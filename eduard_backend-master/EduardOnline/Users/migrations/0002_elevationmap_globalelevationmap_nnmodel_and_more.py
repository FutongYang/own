# Generated by Django 4.2.4 on 2023-08-11 03:18

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ElevationMap',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ot_id', models.IntegerField()),
                ('cost', models.IntegerField()),
                ('creation_date', models.DateField()),
                ('deleted', models.BooleanField()),
            ],
        ),
        migrations.CreateModel(
            name='GlobalElevationMap',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField()),
                ('description', models.TextField()),
                ('cell_size', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='NNModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField()),
                ('description', models.TextField()),
            ],
        ),
        migrations.RemoveField(
            model_name='customuser',
            name='bio',
        ),
        migrations.RemoveField(
            model_name='customuser',
            name='birth_date',
        ),
        migrations.RemoveField(
            model_name='customuser',
            name='location',
        ),
        migrations.AddField(
            model_name='customuser',
            name='ot_token',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='customuser',
            name='registration_date',
            field=models.DateField(default=datetime.datetime.now),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='email',
            field=models.TextField(),
        ),
        migrations.CreateModel(
            name='ReliefMap',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('credit_cost', models.IntegerField()),
                ('elev_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Users.elevationmap')),
                ('model_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Users.nnmodel')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Payments',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('amount', models.IntegerField()),
                ('date', models.DateField()),
                ('no_credits', models.IntegerField()),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AddField(
            model_name='elevationmap',
            name='map_ot_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Users.globalelevationmap'),
        ),
        migrations.AddField(
            model_name='elevationmap',
            name='user_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]