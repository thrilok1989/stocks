# ⚡ Oracle Cloud Quick Start (5 Minutes)

**Get your trading dashboard running on Oracle Cloud in 5 simple steps!**

---

## 🚀 Step 1: Create VM Instance (2 min)

1. Login to [Oracle Cloud Console](https://cloud.oracle.com/)
2. **Compute** → **Instances** → **Create Instance**
3. Settings:
   - Name: `trading-dashboard`
   - Image: `Ubuntu 22.04`
   - Shape: `VM.Standard.E2.1.Micro` (Always Free ✅)
   - ✅ Check "Assign a public IPv4 address"
   - Download SSH key pair
4. Click **Create**

📝 **Note down your Public IP**: `___.___.___.___`

---

## 🔓 Step 2: Open Firewall (1 min)

**In OCI Console:**
1. **Networking** → **VCN** → **Security Lists**
2. **Add Ingress Rule**:
   - Source: `0.0.0.0/0`
   - Port: `5000`
   - Click **Add**

---

## 💻 Step 3: Connect & Setup (1 min)

```bash
# Connect via SSH (replace with your IP and key path)
ssh -i ~/Downloads/ssh-key.key ubuntu@YOUR_PUBLIC_IP

# Update system
sudo apt update && sudo apt install -y python3.11 python3-pip git

# Clone repository
git clone https://github.com/thrilok1989/stocks.git
cd stocks
```

---

## 📦 Step 4: Install & Configure (1 min)

```bash
# Install dependencies
pip3 install -r requirements_flask.txt

# Create environment file
nano .env
```

**Add your credentials:**
```
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_token
```

Save: `Ctrl+X`, `Y`, `Enter`

---

## 🎯 Step 5: Launch! (30 seconds)

```bash
# Start the application
python3 flask_backend.py
```

**✅ Open browser:** `http://YOUR_PUBLIC_IP:5000`

---

## 🔄 Make it Run 24/7 (Optional)

Press `Ctrl+C` to stop, then:

```bash
# Create service
sudo tee /etc/systemd/system/trading-dashboard.service > /dev/null <<EOF
[Unit]
Description=Trading Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/stocks
ExecStart=/usr/bin/python3 /home/ubuntu/stocks/flask_backend.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable & start
sudo systemctl daemon-reload
sudo systemctl enable trading-dashboard
sudo systemctl start trading-dashboard
```

---

## 📊 Useful Commands

```bash
# View logs
sudo journalctl -u trading-dashboard -f

# Restart app
sudo systemctl restart trading-dashboard

# Check status
sudo systemctl status trading-dashboard

# Update code
cd ~/stocks && git pull && sudo systemctl restart trading-dashboard
```

---

## 🆘 Not Working?

### Can't connect to browser?
```bash
# Open OS firewall
sudo ufw allow 5000/tcp
sudo ufw enable
```

### Application crashes?
```bash
# Check logs
sudo journalctl -u trading-dashboard -n 50
```

### Need detailed help?
📖 See [ORACLE_DEPLOYMENT_GUIDE.md](./ORACLE_DEPLOYMENT_GUIDE.md)

---

## 💰 Cost

✅ **100% FREE** with Oracle Always Free Tier!
- 2 VMs (1 OCPU, 1GB RAM each)
- 100 GB storage
- 10 TB monthly traffic

---

**🎉 That's it! Your trading dashboard is live!**

URL: `http://YOUR_PUBLIC_IP:5000`
