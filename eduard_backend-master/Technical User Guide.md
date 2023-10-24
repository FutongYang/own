# Introduction

There are no dogs here.

This is a technical guide for setting up the server and changing essential systems.
There are a number of features that can be added/changed in the system, and it is important to test changes before deploying.

# Deploying server

Refer back to the readme file for a starting guide for getting the system working.
The readme file only details how to install the software, and not how to configure it.
It is assumed that the server is being run on an EC2 instance.

The build created when you follow the readme file steps makes a working debug version of the server that is used for testing.
There are several additional steps that have to be followed before pushing it into service.
## Setting up EC2
To create an EC2 instance, go to Amazon AWS service and follow the instructions here.
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html
Take note of the AWS system name

## Debug menu
Before you deploy the server to production, it is important to turn off the debug menu.
The value for debug is in [settings.py](./EduardOnline/EduardOnline/settings.py)
and it is important to set
~~~
DEBUG = False
~~~
before you deploy the server.


These listed settings must be changed before deployment.
- Google Verification ID
- Debug value
- Default AWS S3 bucket name and keys
- Stripe Keys

## Setup
In the root directory, add a .env file.

## Google Client ID

### Getting Client ID from Google
Follow the instructions on this webpage to create a Google ClientID.

https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid

Since it is a good idea to test on the localhost, do not forget to add
http://localhost
http://localhost:8000
to the authorised Javascript login.

### Installing Client ID
Add to the env file
~~~
CLIENT_ID = 
~~~
with the client ID from above.

## AWS S3 Bucket
The S3 Bucket is where all files are stored, including the .tif elevation models and .png relief maps.
We are using a bucket instead of local storage because it is easier to send files to and from the frontend through a bucket
and because S3 has a built-in CDN network, called Cloudfront.
Cloudfront will make it easier to display images to the end-user as it is a delivery network, meaning that files are not being
passed around through the API calls.
Cloudfront will need to be configured to work with the front end.

### Creating an S3 Bucket
The S3 Bucket is created in the Amazon AWS interface.
https://aws.amazon.com/s3/

Refer to the [pricing guide](https://aws.amazon.com/s3/pricing/) for information on how much S3 costs.

To create the correct S3 bucket, sign into Amazon and create a new bucket called `eotifbucket`. 
For the time being, set up the bucket so that it does not block public access.
Additionally, enable ACLs with the setting "Bucket Owner Preferred".

This is recommended for testing but for deployment enterprise settings will have to be used. 
This is done with "Bucket Policies". These are statements which govern how access to files is determined.
For example, to allow access to the relief file map, use the settings:

~~~
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Statement1",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::eotifbucket/relief/*"
        }
    ]
}
~~~
to allow Cloudfront to access the system.

#### Lifecycles
To clean up data in the eotifbucket, it is recommended that you implement a storage lifecycle
https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html
and expire the documents after 30 days.

#### Setting up Cloudfront
We want to add a CDN to pass images to and from the system.

The easiest way to do this is to setup Cloudfront, Amazon's CDN service.

We used this guide 
https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/GettingStarted.html
to start setting up the Cloudfront storage.

Take note of the Cloudfront url and update the `topography_api.py` file, specifically the `CLOUDFRONT_URL` value.


#### Changing the AWS S3 bucket and keys
To change the AWS S3 bucket, go to the .env file and change the lines
~~~
AWS_S3_ACCESS_KEY_ID  = 
AWS_S3_SECRET_ACCESS_KEY =
AWS_STORAGE_BUCKET_NAME = eotifbucket
~~~
to point to your actual S3 bucket keys.

### Setting up Stripe
To set up stripe, you first create a Stripe business account. Then update the value
~~~
STRIPE_API_KEY=
~~~
to the stripe API key that is listed in Account.
#### Setting up Webhooks
To set up the webhook, go to the Stripe Webhooks page and create a new webhook listener in the Stripe page. Set the webhook listening URL `http://[EC2 IP Address]:8000/api/webhook/`.

Set the webhook to listen to ` checkout.session.completed`. Note the secret, and set
~~~
STRIPE_ENDPOINT_SECRET = 
~~~
to the webhook secret.
## Maintenance

### Querying database
It is important to keep track of all users and their information.
Below is the relational database map.

(MAP GOES HERE)
To get a list of all users, simply query the sqlite3 database inside Django by using VSCode with SQLite. Open the SQLite folder and look inside the CustomUser file.

### Transaction log

There will be an automated transaction log in the future.
The system we are planning on using 

## Adding new features

### Adding extra steps to the model processing pipeline

To add an extra step to the pipeline, navigate to [the image processing files](./EduardOnline/Users/src/).
The file [image.py](./EduardOnline/Users/src/image.py) contains a list of the functions and some test functions at the bottom.
The file [processmodel.py](./EduardOnline/Users/src/processmodel.py) contains the processing pipeline.

It is recommended you test the functions in image.py first before moving on to adding them to the processmodel.py file.
Remember to change the function header when you add any new parameters in both convertDispVal() and runProcess() in [processmodel.py](./EduardOnline/Users/src/processmodel.py).

### Adding new Neural Network models
To add new neural networks, add them to src and add them as models to be loaded in processmodel.py. Then add some extra code in image.py to handle neural networks type 1, 2, and 3 in Contrast.

### Adding new fields to database
The most useful database fields are located in the models file.
To edit them, simply put in their datatype like the others.

### Adding extra fields to the database
Go to the database model, located [in Users](EduardOnline/Users/models.py). Add extra fields, as necessary.
