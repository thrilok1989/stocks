# 🏛️ Oracle Cloud Deployment Guide for NIFTY/SENSEX Trading Dashboard

This guide covers deploying your Python trading application on **Oracle Cloud Infrastructure (OCI)**.

---

## 📋 Table of Contents

1. [Deployment Options](#deployment-options)
2. [Prerequisites](#prerequisites)
3. [Option 1: Oracle Compute Instance (Recommended)](#option-1-oracle-compute-instance-recommended)
4. [Option 2: Oracle Container Instances](#option-2-oracle-container-instances)
5. [Networking & Security](#networking--security)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Cost Optimization](#cost-optimization)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Deployment Options

### Option 1: **Oracle Compute Instance** (VM) ⭐ RECOMMENDED
- **Best for**: Production deployment
- **Cost**: Free tier available (Always Free VM.Standard.E2.1.Micro)
- **Setup time**: 15-20 minutes
- **Pros**: Full control, easy to manage, auto-restart
- **Cons**: Requires basic Linux knowledge

### Option 2: **Oracle Container Instances**
- **Best for**: Docker-familiar teams
- **Cost**: Pay-as-you-go
- **Setup time**: 30-40 minutes (requires Docker setup)
- **Pros**: Containerized, scalable
- **Cons**: More complex, higher cost

---

## 📦 Prerequisites

### Oracle Cloud Account
1. Sign up for Oracle Cloud: https://cloud.oracle.com/
2. Complete identity verification
3. Navigate to OCI Console

### Required Information
- [ ] DhanHQ Client ID and Access Token
- [ ] (Optional) Telegram Bot Token and Chat ID
- [ ] (Optional) News API Key
- [ ] (Optional) Perplexity API Key

### Tools Needed
- SSH client (Terminal on Mac/Linux, PuTTY on Windows)
- Web browser

---

## 🚀 Option 1: Oracle Compute Instance (Recommended)

### Step 1: Create a Compute Instance

1. **Login to OCI Console**
   - Go to: https://cloud.oracle.com/
   - Navigate to **Compute** → **Instances**

2. **Create Instance**
   - Click **Create Instance**
   - Name: `trading-dashboard`

3. **Choose Image and Shape**
   - **Image**: `Oracle Linux 8` or `Ubuntu 22.04` (Recommended)
   - **Shape**:
     - Free Tier: `VM.Standard.E2.1.Micro` (1 OCPU, 1GB RAM) ✅ Always Free
     - Production: `VM.Standard.E4.Flex` (2 OCPUs, 8GB RAM) - Better performance

4. **Networking**
   - **VCN**: Create new or select existing
   - **Subnet**: Public subnet
   - ✅ **Assign a public IPv4 address** (IMPORTANT!)

5. **Add SSH Keys**
   - **Generate SSH key pair** (OCI will provide download)
   - Or upload your existing public key
   - Save the private key securely!

6. **Boot Volume**
   - Size: `50 GB` (default is enough)

7. **Click**: `Create`

Wait 2-3 minutes for the instance to provision. Note down the **Public IP Address**.

---

### Step 2: Configure Firewall Rules

#### A. Configure Security List (OCI Firewall)

1. Go to **Networking** → **Virtual Cloud Networks**
2. Click your VCN → **Security Lists** → **Default Security List**
3. Click **Add Ingress Rules**

**Add these rules:**

| Source CIDR | Protocol | Destination Port | Description |
|-------------|----------|------------------|-------------|
| 0.0.0.0/0 | TCP | 5000 | Flask Backend |
| 0.0.0.0/0 | TCP | 8501 | Streamlit (Optional) |
| 0.0.0.0/0 | TCP | 22 | SSH Access |
| 0.0.0.0/0 | TCP | 80 | HTTP (Optional) |
| 0.0.0.0/0 | TCP | 443 | HTTPS (Optional) |

#### B. Configure OS Firewall (Inside the VM)

Connect via SSH first (see Step 3), then run:

**For Oracle Linux / CentOS:**
```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

**For Ubuntu:**
```bash
sudo ufw allow 5000/tcp
sudo ufw allow 8501/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

---

### Step 3: Connect to Your Instance

Use the public IP and private key from Step 1:

**Mac/Linux:**
```bash
chmod 400 ~/Downloads/ssh-key-*.key
ssh -i ~/Downloads/ssh-key-*.key opc@<YOUR_PUBLIC_IP>
```

**Windows (PuTTY):**
1. Convert `.key` to `.ppk` using PuTTYgen
2. Open PuTTY
3. Host: `opc@<YOUR_PUBLIC_IP>`
4. Connection → SSH → Auth → Browse for `.ppk` file
5. Click **Open**

> **Default Users:**
> - Oracle Linux: `opc`
> - Ubuntu: `ubuntu`

---

### Step 4: Install Python and Dependencies

Once connected via SSH:

#### A. Update System
```bash
sudo yum update -y  # Oracle Linux/CentOS
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu
```

#### B. Install Python 3.11+
**Oracle Linux:**
```bash
sudo yum install -y python3.11 python3.11-pip git
```

**Ubuntu:**
```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

#### C. Verify Installation
```bash
python3.11 --version  # Should show Python 3.11.x
```

---

### Step 5: Deploy the Application

#### A. Clone Repository (or Upload Files)

**Option A: Clone from GitHub**
```bash
cd ~
git clone https://github.com/thrilok1989/stocks.git
cd stocks
```

**Option B: Upload Files via SCP**
```bash
# From your local machine:
scp -i ~/Downloads/ssh-key-*.key -r /path/to/stocks opc@<YOUR_PUBLIC_IP>:~
```

#### B. Install Dependencies
```bash
cd ~/stocks
pip3.11 install -r requirements_flask.txt
pip3.11 install -r requirements.txt  # For all modules
```

#### C. Set Up Environment Variables
```bash
nano .env
```

Add your credentials:
```bash
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Optional
TELEGRAM_CHAT_ID=your_chat_id  # Optional
NEWSDATA_API_KEY=your_news_api_key  # Optional
PERPLEXITY_API_KEY=your_perplexity_key  # Optional
```

Save: `Ctrl+X`, `Y`, `Enter`

---

### Step 6: Run the Application

#### A. Test Run (Foreground)
```bash
python3.11 flask_backend.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

Test in browser: `http://<YOUR_PUBLIC_IP>:5000`

Press `Ctrl+C` to stop.

#### B. Run as Background Service (Production)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/trading-dashboard.service
```

Add this content:
```ini
[Unit]
Description=NIFTY/SENSEX Trading Dashboard
After=network.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/stocks
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3.11 /home/opc/stocks/flask_backend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

> **Note**: Change `User=opc` to `User=ubuntu` if using Ubuntu

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-dashboard
sudo systemctl start trading-dashboard
```

Check status:
```bash
sudo systemctl status trading-dashboard
```

View logs:
```bash
sudo journalctl -u trading-dashboard -f
```

---

### Step 7: Access Your Application

Open your browser:
```
http://<YOUR_PUBLIC_IP>:5000
```

✅ **You're live on Oracle Cloud!**

---

## 🐳 Option 2: Oracle Container Instances

### Prerequisites
- Docker installed locally
- OCI CLI configured
- Docker image pushed to Oracle Container Registry (OCIR)

### Step 1: Create Dockerfile

Create `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_flask.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements_flask.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "flask_backend.py"]
```

### Step 2: Build and Push to OCIR

```bash
# Login to OCIR
docker login <region>.ocir.io
# Username: <tenancy-namespace>/<oci-username>
# Password: <auth-token>

# Build image
docker build -t trading-dashboard .

# Tag image
docker tag trading-dashboard:latest \
  <region>.ocir.io/<tenancy-namespace>/trading-dashboard:latest

# Push to OCIR
docker push <region>.ocir.io/<tenancy-namespace>/trading-dashboard:latest
```

### Step 3: Create Container Instance

1. Go to **Developer Services** → **Container Instances**
2. Click **Create Container Instance**
3. **Name**: `trading-dashboard-ci`
4. **Shape**: `CI.Standard.E4.Flex` (1 OCPU, 8GB RAM)
5. **Image**: Select from OCIR
6. **Environment Variables**: Add your `.env` values
7. **Port Mapping**: 5000:5000
8. Click **Create**

---

## 🔒 Networking & Security

### SSL/HTTPS Setup (Optional but Recommended)

#### Using Nginx as Reverse Proxy

1. **Install Nginx**
```bash
sudo yum install -y nginx  # Oracle Linux
# OR
sudo apt install -y nginx  # Ubuntu
```

2. **Configure Nginx**
```bash
sudo nano /etc/nginx/conf.d/trading-dashboard.conf
```

Add:
```nginx
server {
    listen 80;
    server_name <YOUR_PUBLIC_IP>;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

3. **Start Nginx**
```bash
sudo systemctl enable nginx
sudo systemctl start nginx
```

4. **Access via HTTP**
```
http://<YOUR_PUBLIC_IP>
```

#### Add SSL with Let's Encrypt (Optional)

```bash
# Install certbot
sudo yum install -y certbot python3-certbot-nginx  # Oracle Linux
# OR
sudo apt install -y certbot python3-certbot-nginx  # Ubuntu

# Get certificate (requires domain name)
sudo certbot --nginx -d yourdomain.com
```

---

## 🔧 Environment Configuration

### Production Checklist

- [ ] Set strong firewall rules (limit SSH to your IP)
- [ ] Use environment variables for secrets (never hardcode)
- [ ] Enable automatic backups
- [ ] Set up monitoring alerts
- [ ] Configure log rotation
- [ ] Test auto-restart on crash

### Recommended Security

1. **Limit SSH Access**
   - Edit Security List to allow SSH only from your IP
   - Use SSH key authentication only (disable password)

2. **Enable Fail2Ban**
```bash
sudo yum install -y fail2ban  # Oracle Linux
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

3. **Regular Updates**
```bash
# Add to crontab
sudo crontab -e
# Add: 0 3 * * 0 yum update -y && systemctl reboot
```

---

## 📊 Monitoring & Maintenance

### View Application Logs
```bash
sudo journalctl -u trading-dashboard -f
```

### Restart Application
```bash
sudo systemctl restart trading-dashboard
```

### Update Application
```bash
cd ~/stocks
git pull origin main  # Or your branch
sudo systemctl restart trading-dashboard
```

### Monitor System Resources
```bash
htop  # Install: sudo yum install -y htop
df -h  # Disk usage
free -h  # Memory usage
```

---

## 💰 Cost Optimization

### Oracle Free Tier (Always Free)
✅ **What's Free Forever:**
- 2 Compute instances (VM.Standard.E2.1.Micro)
- 1 OCPU, 1 GB RAM each
- 100 GB Block Storage
- 10 TB/month outbound data transfer

### Monthly Cost Estimate (Paid Tier)
| Component | Specs | Monthly Cost |
|-----------|-------|--------------|
| VM.Standard.E4.Flex | 2 OCPU, 8GB RAM | ~$30-40 |
| Block Storage | 50 GB | ~$2.50 |
| Network (1TB out) | 1TB | ~$8 |
| **Total** | | **~$40-50/month** |

> **Free Tier is sufficient** for this application unless handling extremely high traffic.

---

## 🔍 Troubleshooting

### Issue 1: Cannot Access via Browser

**Check:**
1. Security List has ingress rule for port 5000
2. OS firewall allows port 5000
3. Application is running: `sudo systemctl status trading-dashboard`
4. Correct public IP address

**Fix:**
```bash
# Check if app is listening
sudo netstat -tlnp | grep 5000

# Check firewall
sudo firewall-cmd --list-all  # Oracle Linux
sudo ufw status  # Ubuntu
```

### Issue 2: Application Crashes

**Check logs:**
```bash
sudo journalctl -u trading-dashboard -n 100 --no-pager
```

**Common causes:**
- Missing `.env` file
- Invalid API credentials
- Python dependency issues

### Issue 3: Out of Memory

**Solution:**
- Upgrade to larger shape (2GB+ RAM recommended)
- Add swap space:
```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Issue 4: DhanHQ API Rate Limits

**Symptoms:**
- API errors in logs
- Missing data

**Solution:**
- DhanHQ limits: 25 req/sec, 250 req/min, 7000 req/day
- The app has built-in caching to reduce API calls
- Check `data_cache_manager.py` settings

---

## 📚 Additional Resources

### Oracle Cloud Documentation
- [OCI Compute Instances](https://docs.oracle.com/en-us/iaas/Content/Compute/home.htm)
- [OCI Networking](https://docs.oracle.com/en-us/iaas/Content/Network/Concepts/overview.htm)
- [OCI Free Tier](https://www.oracle.com/cloud/free/)

### Application Documentation
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - Complete setup guide
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
- [CREDENTIALS_SETUP.md](./CREDENTIALS_SETUP.md) - API setup

---

## 🎯 Quick Reference Commands

```bash
# Start application
sudo systemctl start trading-dashboard

# Stop application
sudo systemctl stop trading-dashboard

# Restart application
sudo systemctl restart trading-dashboard

# View logs
sudo journalctl -u trading-dashboard -f

# Check status
sudo systemctl status trading-dashboard

# Update code
cd ~/stocks && git pull && sudo systemctl restart trading-dashboard

# Check if port is open
sudo netstat -tlnp | grep 5000

# Check system resources
htop
```

---

## ✅ Success Checklist

- [ ] Oracle Compute instance created
- [ ] SSH access working
- [ ] Python 3.11 installed
- [ ] Application code deployed
- [ ] Dependencies installed
- [ ] `.env` file configured with API keys
- [ ] Firewall rules configured (OCI + OS)
- [ ] Application running as systemd service
- [ ] Can access dashboard via browser
- [ ] All 9 tabs loading correctly
- [ ] DhanHQ data fetching successfully

---

## 🆘 Need Help?

1. Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
2. Review application logs
3. Verify Oracle Cloud firewall rules
4. Test API credentials separately
5. Check GitHub issues: https://github.com/thrilok1989/stocks/issues

---

**🎉 Congratulations! Your trading dashboard is now running on Oracle Cloud!**

Access it at: `http://<YOUR_PUBLIC_IP>:5000`
