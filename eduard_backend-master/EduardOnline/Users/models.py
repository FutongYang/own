# Users/models.py

from django.contrib.auth.models import AbstractBaseUser
from django.db import models
from datetime import datetime
import uuid

class CustomUser(AbstractBaseUser):
    registration_date = models.DateField(auto_now_add=True)  # set to the current date only when a user is created
    credits = models.IntegerField(default=0)
    ot_token = models.CharField(max_length=255, default="")  # assuming reasonable length
    email = models.EmailField(max_length=254, unique=True)
    password=None
    last_login=None
    USERNAME_FIELD="email"
    
class Payments(models.Model):
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    # amount = models.IntegerField()
    date = models.DateField(auto_now_add=True)
    no_credits = models.IntegerField()

class ElevationMap(models.Model):
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    file_path = models.TextField()
    creation_date = models.DateField(auto_now_add=True)
    deleted = models.BooleanField(default=False)

class ReliefMap(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    elev_id = models.ForeignKey(ElevationMap, on_delete=models.CASCADE)
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    credit_cost = models.IntegerField()