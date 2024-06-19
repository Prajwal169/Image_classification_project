# forms.py
from django import forms
from multiupload.fields import MultiFileField

class ImageUploadForm(forms.Form):
    images = MultiFileField(min_num=1, max_num=15, max_file_size=1024*1024*5)  # Example: max 10 files, each max 5MB
