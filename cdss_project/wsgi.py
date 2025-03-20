"""
WSGI config for cdss_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cdss_project.settings')

# Collect static files for Vercel deployment
if os.environ.get('VERCEL') == '1':
    from django.core.management import call_command
    try:
        call_command('collectstatic', '--noinput', '--clear')
    except Exception as e:
        print(f"Error collecting static files: {e}")

application = get_wsgi_application()

# Add this for Vercel
app = application