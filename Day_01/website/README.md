# NCERT — National Cyber Emergency Response Team

A responsive front-end website built with HTML5, CSS3, and vanilla JavaScript for the NCERT Practical Assignment #01 (Ubuntu Installation, Nginx Deployment & NCERT Front-End Website).

## Project structure

```
website/
├── index.html
├── css/
│   └── style.css
├── js/
│   └── script.js
├── assets/
│   └── favicon.svg
├── nginx/
│   └── ncert.conf        # sample Nginx server block for port 8080
└── README.md
```

No external dependencies (no CDN, no build step) — everything runs from static files, which keeps it reliable on an offline Ubuntu VM.

## Sections included

- Sticky header with logo, title, and responsive navigation menu
- Hero section with animated network-graph background, stats, and CTAs
- About NCERT
- Departments: Cyber Security, Security Operations Center, Software Development, Governance/Risk/Compliance, Digital Forensics Laboratory
- "How NCERT Functions" — 8-stage incident response timeline
- Contact section with a client-side validated form
- Footer with NCERT identity, copyright, and submission credit line

## Run it locally (quick check before deploying)

Just open `index.html` in a browser, or serve the folder with any static server, e.g.:

```bash
# Python
python3 -m http.server 8080

# Node
npx serve -l 8080
```

Then visit `http://localhost:8080`.

## Deploy with Nginx on Ubuntu (Task 2 & 3)

1. Install Nginx:
   ```bash
   sudo apt update
   sudo apt install nginx -y
   nginx -v
   sudo systemctl status nginx
   ```
2. Copy the site files to the web root:
   ```bash
   sudo mkdir -p /var/www/ncert-website
   sudo cp -r index.html css js assets /var/www/ncert-website/
   ```
3. Add the server block (see `nginx/ncert.conf`):
   ```bash
   sudo cp nginx/ncert.conf /etc/nginx/sites-available/ncert.conf
   sudo ln -s /etc/nginx/sites-available/ncert.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```
4. Open `http://localhost:8080` in the browser — the NCERT website should load.

## Notes

- Contact form is front-end only (no backend) — submission just shows a success message.
