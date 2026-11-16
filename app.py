from flask import Flask, request, render_template, send_from_directory, abort
import cv2
import numpy as np
import math
import os
import re
from uuid import uuid4

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
cv2.setNumThreads(1)

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(app.root_path, 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get('/healthz')
def health():
    return {'status': 'ok'}


@app.errorhandler(413)
def upload_too_large(error):
    return 'Image too large. Please upload an image smaller than 20 MB.', 413

def detect_balls(image_path, return_table=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read this image. Please upload a JPG or PNG image.")
    # Normalize large phone photos so detection thresholds are consistent.
    scale = min(1.0, 1200 / max(img.shape[:2]))
    working = cv2.resize(img, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    min_radius = max(3, int(min(height, width) * 0.006))
    max_radius = max(min_radius + 2, int(min(height, width) * 0.065))

    # Restrict candidates to the largest green/blue felt region when available.
    felt = cv2.inRange(hsv, np.array([30, 55, 35]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table = np.full(gray.shape, 255, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > gray.size * 0.1:
            table[:] = 0
            cv2.fillConvexPoly(table, cv2.convexHull(largest), 255)

    # White, low-saturation regions identify cue-ball candidates independently
    # of Hough's circle ordering. Allow off-white pixels and shaded balls.
    white = cv2.inRange(hsv, np.array([0, 0, 195]), np.array([179, 100, 255]))
    white = cv2.bitwise_and(white, table)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not min_radius <= radius <= max_radius or perimeter == 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        fullness = area / (math.pi * radius ** 2)
        if circularity < 0.65 or fullness < 0.65:
            continue
        # A striped ball may have a large white patch; prefer white covering
        # the entire disk rather than just its brightest patch.
        disk = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(disk, (round(x), round(y)), max(1, round(radius)), 255, -1)
        white_fraction = cv2.mean(white, mask=disk)[0] / 255
        mean_saturation = cv2.mean(hsv[:, :, 1], mask=disk)[0]
        score = circularity * fullness * white_fraction * (1 - mean_saturation / 255) ** 3
        candidates.append((score, (x, y, radius)))
    if not candidates:
        raise ValueError("Could not identify a white cue ball. Try a clearer, well-lit photo of the table.")
    cue = max(candidates, key=lambda candidate: candidate[0])[1]

    # For neutral felt, estimate its color from a ring around the cue ball.
    # Keeping detections inside the felt avoids circles on furniture and rails.
    if np.all(table == 255):
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)
        yy, xx = np.ogrid[:height, :width]
        distance = np.hypot(xx - cue[0], yy - cue[1])
        ring = (distance > cue[2] * 1.8) & (distance < cue[2] * 3)
        if np.any(ring):
            felt_color = np.median(lab[ring], axis=0)
            felt = (np.linalg.norm(lab - felt_color, axis=2) < 28).astype(np.uint8) * 255
            felt = cv2.morphologyEx(felt, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
            contours, _ = cv2.findContours(felt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                hull = cv2.convexHull(max(contours, key=cv2.contourArea))
                if (cv2.contourArea(hull) > gray.size * 0.1
                        and cv2.pointPolygonTest(hull, cue[:2], False) >= 0):
                    table[:] = 0
                    cv2.fillConvexPoly(table, hull, 255)
                    table = cv2.erode(table, np.ones((9, 9), np.uint8))

    # A more sensitive circle pass covers different ball sizes in perspective.
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(6, int(cue[2] * 1.3)), param1=80, param2=14,
        minRadius=max(3, int(cue[2] * 0.55)),
        maxRadius=max(4, int(cue[2] * 1.65))
    )
    others = []
    if circles is not None:
        for x, y, radius in circles[0]:
            if (0 <= int(y) < height and 0 <= int(x) < width
                    and table[int(y), int(x)]
                    and math.hypot(x - cue[0], y - cue[1]) > 0.9 * (cue[2] + radius)
                    and all(math.hypot(x - b[0], y - b[1]) > 0.9 * (radius + b[2])
                            for b in others)):
                others.append((float(x), float(y), float(radius)))
    # The target is still a demo choice: the nearest other detected ball.
    others.sort(key=lambda ball: math.hypot(ball[0] - cue[0], ball[1] - cue[1]))
    balls = np.round(np.array([cue] + others) / scale).astype(int)
    if return_table:
        return balls, working, table, scale
    return balls


def cushion_edge_score(lab, samples, inward, step):
    """Score a felt cushion lip, rejecting the brightward edge of its shadow."""
    profiles = []
    for offset in (-2, -1, 1, 2, 5):
        pixels = np.round(samples + inward * step * offset).astype(int)
        if (np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= lab.shape[1])
                or np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= lab.shape[0])):
            return None
        profiles.append(lab[pixels[:, 1], pixels[:, 0]].astype(float))
    outer = (profiles[0] + profiles[1]) / 2
    inner = (profiles[2] + profiles[3]) / 2
    felt = profiles[4]
    # Lab chroma compares the fabric color while tolerating brightness changes.
    chroma_difference = np.linalg.norm(outer[:, 1:] - felt[:, 1:], axis=1)
    drop = outer[:, 0] - inner[:, 0]
    valid = ((chroma_difference < 20)
             & (outer[:, 0] > felt[:, 0] * 0.60)
             & (drop > 2))
    if valid.mean() < 0.55:
        return None
    # A shadow ending on the felt has the opposite brightness transition.
    return float(np.median(drop[valid]) + 20 * valid.mean()
                 - 0.5 * np.median(chroma_difference[valid]))


def refine_cushion_line(lab, line, start, end, inward):
    """Wiggle both endpoints to maximize sharp felt-lip support along the rail."""
    point, direction = line
    # Project the estimated corners onto the selected infinite line.
    a = point + np.dot(start - point, direction) * direction
    b = point + np.dot(end - point, direction) * direction
    length = np.linalg.norm(b - a)
    fractions = np.linspace(0.08, 0.92, 100)
    # Corners and the central pocket gap should not pull the fitted line.
    fractions = fractions[np.abs(fractions - 0.5) > 0.07]
    groups = np.digitize(fractions, [0.35, 0.65])
    step = max(2, length * 0.005)

    def score_line(first, last):
        samples = first + fractions[:, None] * (last - first)
        profiles = []
        for offset in (-2 * step, -step, -1.2, 1.2, step, 2 * step, 5 * step):
            positions = samples + offset * inward
            if (np.any(positions[:, 0] < 1) or np.any(positions[:, 0] >= lab.shape[1] - 1)
                    or np.any(positions[:, 1] < 1) or np.any(positions[:, 1] >= lab.shape[0] - 1)):
                return -np.inf
            values = cv2.remap(lab, positions[:, 0].astype(np.float32)[None, :],
                               positions[:, 1].astype(np.float32)[None, :],
                               cv2.INTER_LINEAR)[0].astype(float)
            profiles.append(values)
        outer = (profiles[0] + profiles[1]) / 2
        inner = (profiles[4] + profiles[5]) / 2
        chroma = np.linalg.norm(outer[:, 1:] - profiles[6][:, 1:], axis=1)
        valid = ((outer[:, 0] - inner[:, 0] > 2) & (chroma < 20)
                 & (outer[:, 0] > profiles[6][:, 0] * 0.6))
        # Measure the sharp transition at the line, rather than the broad
        # contrast somewhere in the cushion/shadow neighborhood.
        sharpness = np.linalg.norm(profiles[2] - profiles[3], axis=1)
        sharpness = np.where(valid, np.minimum(sharpness, 80), 0)
        scores = [np.mean(sharpness[groups == group]) for group in range(3)]
        # Equal support at both ends prevents a good middle from hiding drift.
        return float(np.mean(scores) + 0.5 * min(scores))

    best_offsets = (0.0, 0.0)
    best_score = score_line(a, b)
    radius = max(8, length * 0.025)
    for span, count in ((radius, 17), (radius / 8, 9), (radius / 32, 7)):
        origin = best_offsets
        for da in np.linspace(origin[0] - span, origin[0] + span, count):
            for db in np.linspace(origin[1] - span, origin[1] + span, count):
                if abs(da) > radius or abs(db) > radius:
                    continue
                score = score_line(a + da * inward, b + db * inward)
                if score > best_score:
                    best_score, best_offsets = score, (da, db)
    first = a + best_offsets[0] * inward
    last = b + best_offsets[1] * inward
    return first, (last - first) / np.linalg.norm(last - first)


def detect_rails(table, scale, image=None):
    """Fit four straight cushion edges and intersect them into table corners."""
    if np.all(table == 255):
        return []
    contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    boundary = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(boundary)
    perimeter = cv2.arcLength(hull, True)
    corners = None
    for epsilon in np.linspace(0.01, 0.10, 40):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            corners = approx[:, 0].astype(np.float32)
            break
    if corners is None:
        return []
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)

    # Fit the central part of each rail, avoiding curved pocket cutouts.
    points = boundary[:, 0].astype(np.float32)
    segments = None
    if image is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
        segments = cv2.HoughLinesP(edges, 1, np.pi / 720, 60,
                                   minLineLength=max(30, int(perimeter * 0.06)),
                                   maxLineGap=20)
    lines = []
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        unit = direction / length
        inward = np.array([-unit[1], unit[0]])
        if np.dot(center - (start + end) / 2, inward) < 0:
            inward = -inward
        relative = points - start
        position = relative @ unit / length
        distance = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        selected = points[(position > 0.12) & (position < 0.88)
                          & (distance < max(5, length * 0.025))]
        if len(selected) >= 10:
            vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x, y]), np.array([vx, vy])))
        else:
            lines.append((start, unit))
        # Prefer a felt-colored cushion lip with a darkward inward transition.
        # The edge where the shadow ends instead transitions toward brightness.
        options = []
        if segments is not None:
            for segment in segments[:, 0]:
                a, b = segment[:2].astype(float), segment[2:].astype(float)
                vector = b - a
                segment_length = np.linalg.norm(vector)
                segment_unit = vector / segment_length
                midpoint = (a + b) / 2
                offset = midpoint - start
                along = np.dot(offset, unit) / length
                separation = abs(offset[0] * unit[1] - offset[1] * unit[0])
                if (abs(np.dot(segment_unit, unit)) > 0.995
                        and segment_length > length * 0.22
                        and 0.1 < along < 0.9 and separation < length * 0.08):
                    samples = (a[None, :] + np.linspace(0.15, 0.85, 15)[:, None] * vector
                               + inward * max(5, length * 0.018))
                    pixels = np.round(samples).astype(int)
                    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < table.shape[1])
                             & (pixels[:, 1] >= 0) & (pixels[:, 1] < table.shape[0]))
                    if valid.mean() < 0.9:
                        continue
                    pixels = pixels[valid]
                    if np.mean(table[pixels[:, 1], pixels[:, 0]] > 0) < 0.85:
                        continue
                    edge_samples = a[None, :] + np.linspace(0.15, 0.85, 25)[:, None] * vector
                    score = cushion_edge_score(lab, edge_samples, inward,
                                               max(2, length * 0.005))
                    if score is not None:
                        options.append((score + 0.01 * segment_length - separation * 0.1,
                                        a, segment_unit))
        # If Hough missed the lip, search parallel offsets around the fitted
        # felt boundary using the same fabric and shadow-direction checks.
        if image is not None and not options:
            samples = start + np.linspace(0.15, 0.85, 35)[:, None] * (end - start)
            for shift in np.linspace(-length * 0.06, length * 0.025, 50):
                shifted = samples + shift * inward
                score = cushion_edge_score(lab, shifted, inward, max(2, length * 0.005))
                if score is not None:
                    options.append((score - abs(shift) * 0.1, start + shift * inward, unit))
        if options:
            _, point, direction = max(options, key=lambda item: item[0])
            lines[-1] = (point, direction)
        if image is not None:
            lines[-1] = refine_cushion_line(lab, lines[-1], start, end, inward)
    fitted = []
    for i in range(4):
        p, u = lines[(i - 1) % 4]
        q, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < 1e-5:
            return []
        point = p + np.linalg.solve(matrix, q - p)[0] * u
        if np.linalg.norm(point - corners[i]) > perimeter * 0.08:
            return []
        fitted.append(point)
    return np.asarray(fitted, dtype=np.float32) / scale


def detect_pockets(rails, image=None, scale=1.0):
    """Place pockets at four rail intersections and two long-rail midpoints."""
    if len(rails) != 4:
        return []
    corners = np.asarray(rails, dtype=np.float32)
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    transform = cv2.getPerspectiveTransform(square, corners)
    midpoints = [(square[i] + square[(i + 1) % 4]) / 2 for i in range(4)]
    all_midpoints = cv2.perspectiveTransform(np.float32([midpoints]), transform)[0]
    side_pair = (1, 3) if lengths[[1, 3]].mean() > lengths[[0, 2]].mean() else (0, 2)
    if image is not None:
        # Perspective can reverse apparent rail lengths. Look for the actual
        # pocket gaps on each opposite pair, on the felt-facing side only.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_corners = corners * scale
        center = scaled_corners.mean(axis=0)
        evidence = []
        for i, midpoint in enumerate(all_midpoints * scale):
            direction = scaled_corners[(i + 1) % 4] - scaled_corners[i]
            length = np.linalg.norm(direction)
            unit = direction / length
            inward = np.array([-unit[1], unit[0]])
            if np.dot(center - midpoint, inward) < 0:
                inward = -inward
            radius = max(5, length * 0.055)
            along, depth = np.meshgrid(np.linspace(-length * 0.18, length * 0.18, 101),
                                      np.linspace(-radius * 0.7, radius * 0.25, 15))
            pixels = np.round(midpoint + along[..., None] * unit
                              + depth[..., None] * inward).astype(int)
            valid = ((pixels[..., 0] >= 0) & (pixels[..., 0] < gray.shape[1])
                     & (pixels[..., 1] >= 0) & (pixels[..., 1] < gray.shape[0]))
            values = np.full(along.shape, 255, dtype=float)
            values[valid] = gray[pixels[..., 1][valid], pixels[..., 0][valid]]
            # A pocket is a localized break along the rail. Uniform dark
            # frame edges or rail shadows must not count as a pocket gap.
            darkness = np.mean(values < 70, axis=0)
            localized = np.convolve(darkness, np.ones(9) / 9, mode='valid')
            evidence.append(float(localized.max() - np.median(darkness)))
        scores = [evidence[0] + evidence[2], evidence[1] + evidence[3]]
        side_pair = (0, 2) if scores[0] >= scores[1] else (1, 3)
    projected = all_midpoints[list(side_pair)]
    return [tuple(np.round(point).astype(int)) for point in list(corners) + list(projected)]


def segment_distance(point, start, end):
    vector = end - start
    fraction = np.clip(np.dot(point - start, vector) / max(np.dot(vector, vector), 1e-12), 0, 1)
    return float(np.linalg.norm(point - (start + fraction * vector)))


def easiest_shot(balls, rails, pockets):
    """Rank unobstructed direct pots on an approximately rectified 2:1 table."""
    if len(rails) != 4 or len(pockets) != 6 or len(balls) < 2:
        return None
    corners = np.asarray(rails, np.float32)
    # The detected side pockets identify the physical long rails, regardless
    # of their apparent lengths in the photograph.
    pocket = np.asarray(pockets[4], dtype=float)
    long_edge = min(range(4), key=lambda i: segment_distance(
        pocket, corners[i], corners[(i + 1) % 4]))
    width, height = (2.0, 1.0) if long_edge % 2 == 0 else (1.0, 2.0)
    rectangle = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(corners, rectangle)
    inverse = cv2.getPerspectiveTransform(rectangle, corners)

    def project(points, matrix):
        return cv2.perspectiveTransform(np.float32([points]), matrix)[0].astype(float)

    centers = project(np.asarray(balls)[:, :2], transform)
    holes = project(pockets, transform)
    # Convert detected image radii to table units. Equal physical ball sizes
    # make the median more reliable than an individual noisy circle radius.
    radii = []
    for ball, center in zip(balls, centers):
        x, y, radius = map(float, ball)
        offsets = project([[x + radius, y], [x - radius, y],
                           [x, y + radius], [x, y - radius]], transform)
        radii.append(np.mean(np.linalg.norm(offsets - center, axis=1)))
    radius = float(np.median(radii))
    if not 0 < radius < min(width, height) * 0.1:
        return None
    cue = centers[0]
    best = None
    for index, target in enumerate(centers[1:], start=1):
        for pocket_index, hole in enumerate(holes):
            travel = hole - target
            object_distance = np.linalg.norm(travel)
            if object_distance < radius * 2:
                continue
            outgoing = travel / object_distance
            # Cue-ball center must reach this ghost-ball contact position.
            ghost = target - outgoing * (2 * radius)
            if not (radius <= ghost[0] <= width - radius
                    and radius <= ghost[1] <= height - radius):
                continue
            incoming = ghost - cue
            cue_distance = np.linalg.norm(incoming)
            if cue_distance < radius:
                continue
            cosine = np.clip(np.dot(incoming / cue_distance, outgoing), -1, 1)
            cut_angle = math.degrees(math.acos(cosine))
            if cut_angle > 70:
                continue
            clearance = float('inf')
            for other_index, obstacle in enumerate(centers):
                if other_index in (0, index):
                    continue
                clearance = min(clearance,
                                segment_distance(obstacle, cue, ghost) - 2 * radius,
                                segment_distance(obstacle, target, hole) - 2 * radius)
            if clearance < radius * 0.15:
                continue
            # Favor fuller hits and shorter cue/object travel, with some
            # preference for generous clearance rather than near misses.
            score = (2.5 * (cut_angle / 70) ** 2 + cue_distance / 2
                     + object_distance + 0.2 * radius / max(clearance, radius * 0.15))
            if best is None or score < best['score']:
                best = dict(ball_index=index, pocket_index=pocket_index,
                            ghost=tuple(np.round(project([ghost], inverse)[0]).astype(int)),
                            cut_angle=cut_angle, score=score)
    return best


def top_down_table(balls, rails, pockets):
    """Map detections to an 800 x 400 playing surface, preserving ball IDs."""
    if len(rails) != 4 or len(pockets) != 6:
        return None
    corners = np.asarray(rails, dtype=np.float32)
    long_edge = min(range(4), key=lambda i: segment_distance(
        np.asarray(pockets[4], dtype=float), corners[i], corners[(i + 1) % 4]))
    # Rotate the rectification so the physical long rails are horizontal.
    destination = np.float32([[0, 0], [800, 0], [800, 400], [0, 400]])
    if long_edge % 2:
        destination = np.roll(destination, 1, axis=0)
    transform = cv2.getPerspectiveTransform(corners, destination)

    def project(points):
        return cv2.perspectiveTransform(np.float32([points]), transform)[0]

    centers = project(np.asarray(balls)[:, :2])
    radii = []
    for ball, center in zip(balls, centers):
        x, y, radius = map(float, ball)
        offsets = project([[x + radius, y], [x - radius, y],
                           [x, y + radius], [x, y - radius]])
        radii.append(np.mean(np.linalg.norm(offsets - center, axis=1)))
    # Balls have equal physical sizes in the rectified view.
    radius = float(np.median(radii)) if radii else 8.0
    side_edges = (0, 2) if long_edge % 2 == 0 else (1, 3)
    # Keep P5/P6 ordering consistent with the photographed view.
    sides = [(destination[i] + destination[(i + 1) % 4]) / 2 for i in side_edges]
    return dict(width=800, height=400, radius=radius,
                balls=[dict(x=float(p[0]), y=float(p[1])) for p in centers],
                pockets=[dict(x=float(p[0]), y=float(p[1]))
                         for p in list(destination) + sides])


def cue_view_overlap(angle, side=1):
    """Orthographic cue-direction view; overlap means horizontal width, not area."""
    if angle is None:
        return None
    offset = float(np.clip(math.sin(math.radians(angle)), 0, 1))
    return dict(offset=offset * (1 if side >= 0 else -1),
                overlap=max(0.0, 1 - offset), feasible=angle <= 90)


def enumerate_shots(balls, pockets, overhead=None):
    """Keep photo overlay positions, using rectified coordinates for angles."""
    shots = []
    angle_balls = (np.array([[b['x'], b['y']] for b in overhead['balls']])
                   if overhead is not None else np.asarray(balls)[:, :2])
    angle_pockets = (np.array([[p['x'], p['y']] for p in overhead['pockets']])
                     if overhead is not None else np.asarray(pockets))
    cue = np.asarray(angle_balls[0], dtype=float)
    for target_index, ball in enumerate(balls[1:], start=1):
        target = np.asarray(ball[:2], dtype=float)
        angle_target = np.asarray(angle_balls[target_index], dtype=float)
        incoming = angle_target - cue
        for pocket_index, pocket in enumerate(pockets):
            outgoing = np.asarray(angle_pockets[pocket_index], dtype=float) - angle_target
            denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            angle = None if denominator < 1e-9 else math.degrees(math.acos(
                float(np.clip(np.dot(incoming, outgoing) / denominator, -1, 1))))
            ghost = None
            ghost_angle = None
            overlap = None
            cue_distance = None
            target_distance = np.linalg.norm(outgoing)
            if target_distance > 1e-9:
                # The equal-size ghost cue ball touches the target on the side
                # opposite the pocket: center separation equals two radii.
                separation = (2 * overhead['radius'] if overhead is not None
                              else float(balls[0][2] + ball[2]))
                ghost = angle_target - outgoing / target_distance * separation
                cue_direction = ghost - cue
                cue_distance = float(np.linalg.norm(cue_direction))
                ghost_denominator = np.linalg.norm(cue_direction) * target_distance
                if ghost_denominator > 1e-9:
                    ghost_angle = math.degrees(math.acos(float(np.clip(
                        np.dot(cue_direction, outgoing) / ghost_denominator, -1, 1))))
                    side = cue_direction[0] * outgoing[1] - cue_direction[1] * outgoing[0]
                    overlap = cue_view_overlap(ghost_angle, side)
            label = target + (np.asarray(pocket, dtype=float) - target) * 0.28
            shots.append(dict(number=len(shots) + 1, target=target_index,
                              pocket=pocket_index + 1, angle=angle, ghost_angle=ghost_angle,
                              cue_view=overlap,
                              cue_distance=cue_distance, target_distance=float(target_distance),
                              total_distance=(cue_distance + float(target_distance)
                                              if cue_distance is not None else None),
                              ghost_x=float(ghost[0]) if ghost is not None else None,
                              ghost_y=float(ghost[1]) if ghost is not None else None,
                              x=int(target[0]), y=int(target[1]),
                              px=int(pocket[0]), py=int(pocket[1]),
                              label_x=float(label[0]), label_y=float(label[1])))
    return shots


def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    upload_id = uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f'{upload_id}_input.jpg')
    output_filename = f'{upload_id}_output.jpg'
    file.save(input_path)

    try:
        circles, working, table, scale = detect_balls(input_path, return_table=True)
        rails = detect_rails(table, scale, working)
        pockets = detect_pockets(rails, working, scale)
        img = cv2.imread(input_path)
    except ValueError as error:
        return str(error), 400
    finally:
        # The original is no longer needed once loaded; keep only the result.
        if os.path.exists(input_path):
            os.remove(input_path)

    if len(circles) < 2:
        return "Need at least two balls detected for calculation. Try another image.", 400

    # Detection returns the white cue ball first, then other balls by distance.
    cue_ball = (circles[0][0], circles[0][1])
    shot = easiest_shot(circles, rails, pockets)
    angle = calculate_angle(cue_ball, shot['ghost']) if shot else None

    # Draw circles and line on the image
    overhead = top_down_table(circles, rails, pockets)
    possible_shots = enumerate_shots(circles, pockets, overhead) if overhead else []

    # Cyan straight lines show the four fitted inner rails.
    if len(rails) == 4:
        cv2.polylines(img, [np.round(rails).astype(np.int32)], True,
                      (255, 255, 0), 12, cv2.LINE_AA)

    # Draw cue ball in green
    cv2.circle(img, cue_ball, circles[0][2], (0,255,0), 15)
    # Draw every other detected ball in red, using the same bold outline.
    for index, (x, y, radius) in enumerate(circles[1:], start=1):
        color = (255, 0, 255) if shot and index == shot['ball_index'] else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), int(radius), color, 15)
    if shot:
        target_ball = tuple(map(int, circles[shot['ball_index']][:2]))
        target_radius = int(circles[shot['ball_index']][2])
        cv2.drawMarker(img, target_ball, (255, 0, 255), cv2.MARKER_CROSS,
                       max(3, round(target_radius * 1.2)),
                       max(1, min(15, target_radius // 4)), cv2.LINE_AA)
        cv2.line(img, cue_ball, shot['ghost'], (255, 0, 0), 8, cv2.LINE_AA)
        cv2.line(img, target_ball, pockets[shot['pocket_index']], (255, 0, 255), 8, cv2.LINE_AA)

    # Yellow crosses distinguish pocket locations from ball outlines.
    marker_size = max(24, round(max(img.shape[:2]) * 0.012))
    for pocket in pockets:
        cv2.drawMarker(img, pocket, (0, 255, 255), cv2.MARKER_CROSS,
                       marker_size, 15, cv2.LINE_AA)

    # Put angle text on image
    cv2.putText(
        img, f'Shoot Angle: {angle:.1f} deg' if shot else 'No clear direct pocket shot found',
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )

    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    if not cv2.imwrite(output_path, img):
        raise RuntimeError("Could not save the analyzed image")

    # Render template with result image and angle
    return render_template('results.html', angle=angle, shot=shot, image_file=output_filename,
                           possible_shots=possible_shots, overhead=overhead, circles=circles.tolist(),
                           pockets=pockets, image_width=img.shape[1], image_height=img.shape[0],
                           overlay_font=max(12, img.shape[1] * 0.015),
                           overlay_stroke=max(2, img.shape[1] * 0.0015))

@app.route('/uploads/<filename>')
def send_image(filename):
    # Serve generated results only, never raw uploads or legacy shared images.
    if not re.fullmatch(r'[0-9a-f]{32}_output\.jpg', filename):
        abort(404)
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
