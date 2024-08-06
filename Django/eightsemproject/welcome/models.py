from django.db import models



class Rand(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.name

class StudentInfo(models.Model):
    name = models.CharField(max_length=255)
    image_data = models.TextField()  # Store the base64 image data as text
    vector = models.BinaryField()    # Store the feature vector as binary data
    semester = models.IntegerField(default=1)

    def __str__(self):
        return self.name