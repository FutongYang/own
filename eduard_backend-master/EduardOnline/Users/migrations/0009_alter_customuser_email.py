# Generated by Django 4.2.3 on 2023-08-29 06:01

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("Users", "0008_customuser_alter_elevationmap_user_id_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="customuser",
            name="email",
            field=models.EmailField(max_length=254, unique=True),
        ),
    ]
