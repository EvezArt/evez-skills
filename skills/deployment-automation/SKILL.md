---
name: deployment-automation
description: One-command deployment to Vercel, Railway, Netlify, and Heroku
---

# deployment-automation

**One-command deployment to Vercel, Railway, Netlify, and Heroku**

---

## Overview

Automate deployments to popular hosting platforms with a single command. Detects platform configuration, installs required CLIs, and handles deployment workflow automatically.

**Time Saved**: ~30 minutes per deployment  
**Complexity**: Medium  
**Prerequisites**: Node.js, npm, git

---

## When to Use

- Deploying web applications to production
- Setting up CI/CD pipelines
- Switching between hosting platforms
- Automating deployment workflows
- Teaching deployment best practices

---

## Supported Platforms

1. **Vercel** - Next.js, React, static sites
2. **Railway** - Full-stack apps, databases, backend
3. **Netlify** - Static sites, JAMstack
4. **Heroku** - Traditional web apps, APIs

---

## Quick Start

### 1. Deploy a Project

```bash
python3 /home/ubuntu/skills/deployment-automation/scripts/deploy.py /path/to/project
```

### 2. Deploy Preview/Staging

```bash
python3 /home/ubuntu/skills/deployment-automation/scripts/deploy.py /path/to/project --preview
```

---

## How It Works

### Phase 1: Detection
- Scans project for platform configuration files
- Identifies: `vercel.json`, `railway.toml`, `netlify.toml`, `Procfile`
- Checks `package.json` for custom deploy scripts

### Phase 2: CLI Setup
- Checks if platform CLI is installed
- Offers to install if missing
- Verifies authentication

### Phase 3: Deployment
- Runs platform-specific deploy command
- Handles production vs preview modes
- Reports success/failure

---

## Usage Examples

### Example 1: Deploy Next.js App to Vercel

```bash
# Project has vercel.json
python3 /home/ubuntu/skills/deployment-automation/scripts/deploy.py ~/my-nextjs-app

# Output:
# 📁 Project: /home/ubuntu/my-nextjs-app
# 🎯 Mode: Production
# ✅ Detected platforms: vercel
# 🚀 Deploying to Vercel...
# ✅ Deployment successful!
```

### Example 2: Deploy Full-Stack App to Railway

```bash
# Project has railway.toml
python3 /home/ubuntu/skills/deployment-automation/scripts/deploy.py ~/my-fullstack-app

# Output:
# 📁 Project: /home/ubuntu/my-fullstack-app
# 🎯 Mode: Production
# ✅ Detected platforms: railway
# 🚀 Deploying to Railway...
# ✅ Deployment successful!
```

### Example 3: Deploy Preview

```bash
# Deploy to preview/staging environment
python3 /home/ubuntu/skills/deployment-automation/scripts/deploy.py ~/my-app --preview

# Output:
# 📁 Project: /home/ubuntu/my-app
# 🎯 Mode: Preview
# ✅ Detected platforms: vercel
# 🚀 Deploying to Vercel...
# ✅ Deployment successful!
```

---

## Configuration Templates

### Vercel (`vercel.json`)

Use template: `/home/ubuntu/skills/deployment-automation/templates/vercel.json`

```json
{
  "version": 2,
  "name": "my-project",
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "env": {
    "DATABASE_URL": "@database_url"
  }
}
```

### Railway (`railway.toml`)

Use template: `/home/ubuntu/skills/deployment-automation/templates/railway.toml`

```toml
[build]
builder = "NIXPACKS"
buildCommand = "npm run build"

[deploy]
startCommand = "npm start"
```

---

## Platform Selection Guide

**Choose Vercel If**:
- Using Next.js or React
- Need fast global CDN
- Want preview deployments
- Deploying frontend only

**Choose Railway If**:
- Need database included
- Building full-stack app
- Want simple pricing
- Need backend services

**Choose Netlify If**:
- Building static site
- Using JAMstack
- Need forms handling

**Choose Heroku If**:
- Using traditional frameworks
- Need extensive add-ons
- Want mature ecosystem

---

## Environment Variables

### Set Variables

**Vercel**:
```bash
vercel env add DATABASE_URL
vercel env pull
```

**Railway**:
```bash
railway variables set DATABASE_URL=postgres://...
railway variables
```

**Netlify**:
```bash
netlify env:set DATABASE_URL postgres://...
netlify env:list
```

**Heroku**:
```bash
heroku config:set DATABASE_URL=postgres://...
heroku config
```

---

## Troubleshooting

### Build Fails
1. Check build logs
2. Verify dependencies in `package.json`
3. Test build locally: `npm run build`
4. Check Node version compatibility

### Deploy Succeeds But Site Broken
1. Check runtime logs
2. Verify environment variables set
3. Test API endpoints
4. Check database connection

### CLI Not Installed
Script will prompt to install automatically:
```
⚠️  vercel CLI not installed
Install vercel CLI? (y/n): y
📦 Installing vercel CLI...
```

### Authentication Required
```bash
# Vercel
vercel login

# Railway
railway login

# Netlify
netlify login

# Heroku
heroku login
```

---

## Best Practices

1. **Test Locally First**
   ```bash
   npm run build
   npm start
   ```

2. **Use Preview Deployments**
   ```bash
   deploy.py /path/to/project --preview
   ```

3. **Set Up Health Checks**
   ```javascript
   // pages/api/health.js
   export default function handler(req, res) {
     res.status(200).json({ status: 'ok' });
   }
   ```

4. **Monitor Logs**
   ```bash
   vercel logs
   railway logs
   netlify logs
   heroku logs --tail
   ```

5. **Version Control**
   ```bash
   git tag v1.0.0
   git push --tags
   ```

---

## Integration with CI/CD

### GitHub Actions

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy
        run: |
          python3 skills/deployment-automation/scripts/deploy.py .
```

---

## Advanced Usage

### Custom Deploy Script

If project has `"deploy"` script in `package.json`:

```json
{
  "scripts": {
    "deploy": "custom-deploy-command"
  }
}
```

The skill will detect and use it automatically.

### Multiple Platforms

If multiple platforms detected, first one is used:

```
✅ Detected platforms: vercel, netlify
🚀 Deploying to Vercel...
```

To use different platform, remove unwanted config files.

---

## Files

### Scripts
- `scripts/deploy.py` - Main deployment automation script

### Templates
- `templates/vercel.json` - Vercel configuration template
- `templates/railway.toml` - Railway configuration template

### References
- `references/platforms.md` - Complete platform comparison and guide

---

## Related Skills

- **github-workflow-automation** - Automate CI/CD with GitHub Actions
- **error-monitoring-setup** - Set up Sentry for deployed apps
- **database-schema-generator** - Generate database schemas before deployment

---

## Workflow

```
1. User runs deploy.py with project path
2. Script detects platform (vercel.json, railway.toml, etc.)
3. Script checks if CLI installed
4. Script prompts to install if missing
5. Script runs deployment command
6. Script reports success/failure
```

---

## Success Criteria

✅ Platform detected automatically  
✅ CLI installed if needed  
✅ Deployment succeeds  
✅ URL returned  
✅ Logs accessible  

---

## Time Savings

**Manual Deployment**: ~30 minutes
- Install CLI: 5 min
- Configure: 10 min
- Set environment variables: 5 min
- Deploy: 5 min
- Troubleshoot: 5 min

**With This Skill**: ~2 minutes
- Run script: 30 sec
- Automated detection and deployment: 1.5 min

**Savings**: ~28 minutes per deployment

---

## Future Enhancements

- Support for AWS, Azure, GCP
- Automatic database migrations
- Blue-green deployments
- Canary releases
- Automatic rollback on errors
- Deployment notifications (Slack, email)
- Performance monitoring integration

---

**Created**: February 10, 2026  
**Status**: Production Ready  
**Tested**: Vercel, Railway, Netlify, Heroku
