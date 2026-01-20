# Jupyter Lab Configuration for QuantRS2

c = get_config()

# Network configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Security configuration (for development only)
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Enable extensions
c.LabApp.collaborative = True

# Set default notebook directory
c.ServerApp.root_dir = '/home/quantrs'

# Enable autosave
c.FileContentsManager.autosave_interval = 60

# Kernel configuration
c.KernelManager.autorestart = True

# Resource limits
c.MappingKernelManager.cull_idle_timeout = 3600  # 1 hour
c.MappingKernelManager.cull_interval = 300       # 5 minutes

# Logging
c.Application.log_level = 'INFO'