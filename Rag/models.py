from django.db import models

# Create your models here.
# No models needed for this simple chatbot application 

class Patient(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    contact = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name 