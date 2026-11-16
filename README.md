# Pool Shot Assistant

A Flask/OpenCV app for detecting pool balls and rails, showing a corrected 2:1
top-down table, and exploring shot angles, ghost balls, overlaps, and distances.
Detection and shot recommendations are estimates.

## Run locally

```powershell
conda activate app_env
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Debug mode is off by default. For local development,
set `$env:FLASK_DEBUG = '1'` before starting the app.

The server uses `opencv-python-headless` because it does not open desktop windows.
Avoid installing it alongside another OpenCV package in the same environment.
If your existing environment has `opencv-python`, uninstall that package before
installing the requirements. The experimental `test.py` uses desktop windows and
is not part of the web app or automated tests.

## Deploy on Render

1. Commit the application, templates, tests, `requirements.txt`, `.python-version`,
   and `render.yaml`, then push your branch to GitHub.
2. In Render, choose **New > Blueprint** and connect this repository.
3. Select your branch and review the service defined in `render.yaml`. It selects
   a Free Python web service. Choose a paid instance if you need more CPU/memory
   or want to avoid the free service's idle spin-down.
4. Deploy the Blueprint. Once healthy, open the public URL Render provides.

Render reads `.python-version` to use Python 3.11. No API keys or database are
required. Gunicorn imports `app:app`; it does not run Flask's development server.

Alternatively, choose **New > Web Service**, connect the repository, and enter:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 --access-logfile - --error-logfile -` |
| Health check path | `/healthz` |
| Root directory | Leave blank |

The single worker processes one analysis at a time to limit memory use. Large
photos and concurrent visitors can be slow on the Free instance. Check Render's
logs if an analysis times out or the process runs out of memory.

Official references:
- https://render.com/docs/deploy-flask
- https://render.com/docs/blueprint-spec
- https://render.com/docs/python-version
- https://render.com/docs/free

## Uploads and storage

- Upload requests are limited to 20 MB.
- Each request has unique input/result filenames, preventing users from
  overwriting each other's images.
- Originals are deleted after loading, including detection failures.
- Anyone with a result URL can open it; there is no login or per-user access control.
- Results have no scheduled cleanup yet. They remain until removed or the host's
  ephemeral filesystem resets. Render can remove them on restart/redeployment;
  Free instances also lose them on spin-down. Old result links can then return 404.
- `UPLOAD_FOLDER` can override the default `uploads` directory. Multi-instance
  hosting needs an appropriate shared storage design.
- `.gitignore` excludes uploads, generated images, environments, and secrets.
  Files already tracked by Git are still tracked: inspect `git ls-files uploads`
  and remove personal images from Git's index before publishing the repository.

## Automated checks

```powershell
python -m unittest discover -s tests -v
```

Gunicorn runs on Render's Linux servers; it is not intended for local Windows
execution. Test locally with `python app.py`.
