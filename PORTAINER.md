# Deploying with Portainer (Add stack from repository)

Prerequisites
- Portainer with access to your Docker host (able to build images).
- This Git repository reachable from the machine where Portainer runs (public or private with credentials).
- Host `nginx` will act as the TLS reverse-proxy and must be configured separately (example below).

Quick steps (Portainer UI)
1. Open Portainer → Stacks → Add stack.
2. Choose **Repository**.
3. Set the **Repository URL** to this repo (e.g. `https://github.com/<user>/ProbApp.git`).
4. Set **Repository reference** to the branch (e.g. `main`).
5. Set **Compose path** to `docker-compose.yml` (or `portainer-stack.yml` if you create one).
6. Under **Build images** enable build (Portainer will run `docker build` on the host for `build: .`).
7. Click **Deploy the stack**.

Notes
- The repository's `docker-compose.yml` uses `build: .` so Portainer must be allowed to build images on the target Docker engine.
- The `app` service binds to `127.0.0.1:8501` by design so the container is not publicly reachable — host `nginx` proxies traffic to it.
- Do NOT store TLS private keys in the repo. Use host filesystem or Portainer Secrets if you must manage keys.

Host `nginx` example (place in your host `/etc/nginx/sites-available/app.conf`)
```nginx
server {
    listen 80;
    server_name app.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name app.example.com;

    ssl_certificate     /etc/ssl/certs/fullchain.pem;
    ssl_certificate_key /etc/ssl/private/privkey.pem;

    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;
    }
}
```

Verify after deploy
- In Portainer check the stack and container `probapp_streamlit` status and logs.
- On the Docker host run:
```bash
curl -I http://127.0.0.1:8501
```
- After host `nginx` and DNS configured, visit `https://app.example.com`.

Troubleshooting
- If Portainer fails to build:
  - Ensure Docker on the host has enough permissions and build support.
  - Build locally, push to a registry, then update `docker-compose.yml` to use `image: your-registry/probapp:tag` instead of `build: .` and deploy the stack referencing the registry image.
