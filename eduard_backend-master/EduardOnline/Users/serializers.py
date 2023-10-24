from rest_framework import serializers
from .models import CustomUser, ElevationMap, ReliefMap

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ["registration_date", "credits", "ot_token", "email"]
    
class ElevationMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = ElevationMap
        fields = ["user_id", "file_path", "creation_date", "deleted"]

class ReliefMapSerializer(serializers.ModelSerializer):
    class Meta:
        model=ReliefMap
        fields=["elev_id", "user_id", "credit_cost"]

class PurchaseCreditsSerializer(serializers.Serializer):
    amount = serializers.IntegerField()

class LoginSerializer(serializers.Serializer):
    token = serializers.CharField()

class GenerateMapSerializer(serializers.Serializer):
    map_name = serializers.CharField()
    macroDisplayedValue = serializers.IntegerField(default=0)
    microDisplayedValue = serializers.IntegerField(default=0)
    illuminationDisplayedValue = serializers.IntegerField(default=0)
    flatAreasAmountDisplayedValue = serializers.IntegerField(default=0)
    flatAreasSizeDisplayedValue = serializers.IntegerField(default=0)
    terrainTypeDisplayedValue1 = serializers.IntegerField(default=0)
    terrainTypeDisplayedValue2 = serializers.IntegerField(default=100)
    nnType = serializers.IntegerField(default=1)
    aerialPerpectiveDisplayedValue = serializers.IntegerField(default=100)
    contrastDisplayedValue = serializers.IntegerField(default=0)

class DownloadMapSerializer(serializers.Serializer):
    dem_type = serializers.CharField(default="AW3D30")
    south = serializers.FloatField()
    north = serializers.FloatField()
    west = serializers.FloatField()
    east = serializers.FloatField()
    amount = serializers.IntegerField(default = 0)