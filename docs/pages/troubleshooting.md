---
layout: page
title: Troubleshooting Guide
permalink: /pages/troubleshooting/
---

# Troubleshooting Guide

This comprehensive troubleshooting guide helps resolve common issues with the Multi-Agent Spacecraft Docking System.

## 🚨 Quick Diagnostics

### **System Health Check**

Run this first to identify issues:

```bash
# Quick system validation
make validate-quick

# Detailed system information
make system-info

# Check dependencies
python -c "
import sys
print('Python:', sys.version)
try:
    import numpy, scipy, matplotlib
    print('✅ Core dependencies installed')
except ImportError as e:
    print('❌ Missing dependencies:', e)
"
```

### **Common Symptoms**

| Symptom | Quick Fix |
|---------|-----------|
| **Import errors** | `make install` then `export PYTHONPATH=.` |
| **Slow performance** | Reduce agents: `--agents 3`, disable realtime |
| **Memory errors** | Reduce horizon: `--horizon 10`, close other apps |
| **No visualization** | Install GUI: `sudo apt install python3-tk` |
| **Permission errors** | Use `sudo` or check file permissions |

## 📦 Installation Issues

### **Problem: Import Errors**

```
ModuleNotFoundError: No module named 'spacecraft_drmpc'
```

**Solution:**
```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Reinstall package
pip uninstall spacecraft-drmpc
pip install -e .

# Verify installation
python -c "import spacecraft_drmpc; print('✅ Success')"
```

### **Problem: Missing Dependencies**

```
ImportError: No module named 'numpy'
```

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# For specific missing packages
pip install numpy scipy matplotlib cvxpy

# Verify core dependencies
python -c "
import numpy, scipy, matplotlib, cvxpy
print('✅ All core dependencies available')
"
```

### **Problem: Version Conflicts**

```
ERROR: Package 'numpy' requires a different version
```

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv spacecraft_env
source spacecraft_env/bin/activate  # Linux/Mac
# spacecraft_env\Scripts\activate   # Windows

# Fresh install
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### **Problem: Permission Denied**

```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Install for user only
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/

# Check write permissions
ls -la results/
chmod 755 results/
```

## 🖥️ Runtime Issues

### **Problem: Slow Performance**

**Symptoms:**
- Simulation runs much slower than real-time
- High CPU usage (>90%)
- Memory usage increases over time

**Solutions:**
```bash
# Reduce computational load
python main.py --agents 3 --horizon 10 --no-realtime

# Enable performance optimizations
python main.py --performance-mode --parallel

# Profile performance
python -m cProfile -o profile.out main.py --scenario single --duration 60
python -c "
import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative').print_stats(10)
"
```

### **Problem: Memory Errors**

```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Reduce memory usage
python main.py \
  --agents 5 \
  --horizon 10 \
  --history-length 100 \
  --no-save

# Monitor memory usage
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')
print(f'CPU count: {psutil.cpu_count()}')
"

# For large simulations, use chunked processing
python scripts/large_simulation.py --batch-size 5
```

### **Problem: Controller Convergence Issues**

**Symptoms:**
- Agents fail to reach targets
- Oscillatory behavior
- "Optimization infeasible" errors

**Solutions:**
```python
# Adjust controller parameters
config = {
    'prediction_horizon': 15,      # Reduce from 20+
    'time_step': 0.1,             # Increase from 0.05
    'solver_tolerance': 1e-4,     # Relax from 1e-6
    'max_iterations': 50,         # Reduce from 100
    'uncertainty_level': 0.1      # Reduce from 0.3
}

# Check feasibility
controller = DRMPCController(config)
result = controller.check_feasibility(initial_state, target_state)
if not result.feasible:
    print(f"Infeasible: {result.reason}")
```

### **Problem: Communication Failures**

**Symptoms:**
- Agents fail to coordinate
- "Connection timeout" errors
- Formation breaks apart

**Solutions:**
```bash
# Check network connectivity
ping localhost
netstat -an | grep 8080

# Increase timeouts
python main.py --timeout 5000 --retry-attempts 5

# Use reliable communication mode
python main.py --comm-protocol tcp --reliable-mode
```

## 🎨 Visualization Issues

### **Problem: No Display/Blank Plots**

**Linux:**
```bash
# Install GUI libraries
sudo apt-get update
sudo apt-get install python3-tk python3-dev python3-tk-dev

# Set display
export DISPLAY=:0.0

# Test display
python -c "
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.show()
"
```

**macOS:**
```bash
# Install Tkinter
brew install python-tk

# Or use backend that doesn't require display
python -c "
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.savefig('test.png')
print('Plot saved as test.png')
"
```

**Windows:**
```powershell
# Usually Tkinter comes with Python
# If issues, reinstall Python with Tkinter option checked

# Test with different backend
python -c "
import matplotlib
print('Available backends:', matplotlib.rcsetup.all_backends)
matplotlib.use('TkAgg')
"
```

### **Problem: Slow/Jerky Visualization**

**Solutions:**
```python
# Reduce update frequency
viewer_config = {
    'update_rate': 5,         # Hz, reduce from 30
    'plot_history': 100,      # Reduce trail length
    'real_time': False,       # Disable for faster replay
    'quality': 'fast'         # vs 'high'
}

# Use efficient plotting
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
# ... plotting code ...
plt.show()  # Show all at once
```

### **Problem: Missing Plot Elements**

**Solutions:**
```python
# Check data availability
print(f"State history length: {len(agent.state_history)}")
print(f"Control history length: {len(agent.control_history)}")

# Enable data saving
simulator_config = {
    'save_states': True,
    'save_controls': True,
    'save_performance': True
}

# Debug plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
# Add debug prints
print(f"Plotting {len(x_data)} points")
```

## 🐳 Docker Issues

### **Problem: Build Failures**

```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**
```bash
# Check Docker installation
docker --version
docker-compose --version

# Clean Docker cache
docker system prune -af

# Rebuild without cache
docker build --no-cache -t spacecraft-drmpc .

# Check disk space
df -h
```

### **Problem: Container Crashes**

```
Container exits with code 137 (OOMKilled)
```

**Solutions:**
```bash
# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory: 8GB

# Check container logs
docker logs spacecraft-sim

# Run with resource limits
docker run --memory=4g --cpus=2 spacecraft-drmpc

# Use Docker Compose with limits
services:
  spacecraft-sim:
    image: spacecraft-drmpc
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

### **Problem: Volume Mount Issues**

```
PermissionError: [Errno 13] Permission denied: '/app/results'
```

**Solutions:**
```bash
# Fix permissions on host
chmod -R 755 results/
chown -R $USER:$USER results/

# Use proper user in Docker
docker run --user $(id -u):$(id -g) \
  -v $(pwd)/results:/app/results \
  spacecraft-drmpc

# Check mount points
docker run -it spacecraft-drmpc ls -la /app/
```

## 🔒 Security Issues

### **Problem: Certificate Errors**

```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
```bash
# For development, disable SSL verification (NOT for production)
export PYTHONHTTPSVERIFY=0

# Install certificates (macOS)
/Applications/Python\ 3.x/Install\ Certificates.command

# Update certificates (Linux)
sudo apt-get update && sudo apt-get install ca-certificates

# Use custom certificate path
export SSL_CERT_FILE=/path/to/cacert.pem
```

### **Problem: Permission/Access Denied**

```
PermissionError: Access denied to secure communication
```

**Solutions:**
```bash
# Check firewall settings
sudo ufw status
sudo ufw allow 8080

# Run with proper permissions
sudo -E python main.py --secure-mode

# Check port availability
netstat -tulpn | grep 8080
```

## ⚡ Performance Tuning

### **Optimization Strategies**

**For Small Systems (1-5 spacecraft):**
```python
config = {
    'prediction_horizon': 20,
    'control_frequency': 100,
    'real_time': True,
    'parallel_processing': False
}
```

**For Medium Systems (5-15 spacecraft):**
```python
config = {
    'prediction_horizon': 15,
    'control_frequency': 50,
    'real_time': True,
    'parallel_processing': True,
    'batch_processing': True
}
```

**For Large Systems (15+ spacecraft):**
```python
config = {
    'prediction_horizon': 10,
    'control_frequency': 25,
    'real_time': False,
    'parallel_processing': True,
    'distributed_computing': True,
    'hierarchical_control': True
}
```

### **Memory Management**

```python
# Limit history storage
agent_config = {
    'max_history_length': 1000,  # Limit stored states
    'save_frequency': 10,        # Save every 10th step
    'compress_data': True        # Use data compression
}

# Garbage collection
import gc
gc.collect()  # Force garbage collection

# Memory profiling
from memory_profiler import profile

@profile
def run_simulation():
    # Your simulation code
    pass
```

## 🔧 Development Issues

### **Problem: Test Failures**

```
FAILED tests/test_controller.py::test_convergence
```

**Solutions:**
```bash
# Run specific test with verbose output
pytest tests/test_controller.py::test_convergence -v -s

# Debug test
pytest --pdb tests/test_controller.py::test_convergence

# Check test dependencies
pytest --collect-only

# Update test fixtures
pytest --fixtures tests/
```

### **Problem: Import Errors in Development**

**Solutions:**
```bash
# Set Python path for development
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Install in development mode
pip install -e .

# Check sys.path
python -c "import sys; print('\n'.join(sys.path))"
```

### **Problem: Code Quality Issues**

```bash
# Fix formatting
black src/ tests/
isort src/ tests/

# Fix linting issues
flake8 src/ tests/ --max-line-length=88

# Check types
mypy src/ --ignore-missing-imports
```

## 📱 Platform-Specific Issues

### **Windows Specific**

**Long Path Issues:**
```powershell
# Enable long paths in Windows
git config --system core.longpaths true

# Use PowerShell as administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path Separator Issues:**
```python
import os
config_path = os.path.join('config', 'mission.yaml')  # Cross-platform
```

### **macOS Specific**

**M1/M2 Chip Issues:**
```bash
# Use Rosetta if needed
arch -x86_64 python main.py

# Install dependencies for Apple Silicon
pip install --upgrade --force-reinstall numpy scipy
```

### **Linux Specific**

**GUI Issues on Headless Servers:**
```bash
# Use virtual display
sudo apt-get install xvfb
xvfb-run -a python main.py --visualize

# Or use non-interactive backend
export MPLBACKEND=Agg
```

## 🆘 Getting Additional Help

### **Diagnostic Information to Include**

When reporting issues, include:

```bash
# System information
python --version
pip --version
uname -a  # Linux/Mac
systeminfo  # Windows

# Package versions
pip list | grep -E "(numpy|scipy|matplotlib|cvxpy)"

# Error reproduction
# Minimal example that reproduces the issue

# Log files
cat logs/system.log | tail -50
```

### **Support Channels**

1. **📚 Documentation**: Check [technical docs](technical-documentation.md) first
2. **💬 GitHub Issues**: Create [detailed issue report](https://github.com/your-repo/issues)
3. **💼 Discussions**: Join [community discussion](https://github.com/your-repo/discussions)
4. **📧 Email**: Technical support at spacecraft-drmpc@example.com

### **Creating Good Bug Reports**

**Template:**
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Install spacecraft-drmpc v2.1.0
2. Run `python main.py --scenario formation --agents 5`
3. Error occurs after 30 seconds

## Expected Behavior
Formation should maintain stable configuration

## Actual Behavior
Agents diverge and collision warnings appear

## Environment
- OS: Ubuntu 22.04
- Python: 3.11.5
- spacecraft-drmpc: 2.1.0
- RAM: 16GB
- CPU: Intel i7-12700K

## Logs
```
[Include relevant log output]
```

## Additional Context
Any other relevant information
```

---

## 🔍 Advanced Debugging

### **Enable Debug Logging**

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# For specific modules
logging.getLogger('spacecraft_drmpc.controllers').setLevel(logging.DEBUG)
```

### **Performance Profiling**

```bash
# CPU profiling
python -m cProfile -o profile.stats main.py --scenario single --duration 60

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling
pip install memory_profiler
python -m memory_profiler main.py --scenario formation --duration 30
```

### **Network Debugging**

```bash
# Monitor network traffic
sudo tcpdump -i any port 8080

# Check active connections
netstat -an | grep 8080

# Test connectivity
telnet localhost 8080
```

This troubleshooting guide covers the most common issues. For specific problems not covered here, please check our [GitHub Issues](https://github.com/your-repo/spacecraft-drmpc/issues) or create a new issue with detailed information.