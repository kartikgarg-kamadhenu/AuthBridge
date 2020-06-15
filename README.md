About the Api : 
It is a flask api that accepts image byte string as input along with the side of the passport which is front or back  in a json format and returns back the cropped and aligned picture of the passport as a image byte string only in the json format.
Setup:
 app.py
 tests.py
 image_to_bytes.py
 crop_f
 new_img
 front_r
 back_r
  project
	 config
		 config.ini
	 functions
		 functions.py
	 routes
		 routes.py
	 uploads
	 __init__.py
app.py : It is the file that runs the api

view.py : It has the routes of the api

config.ini : There is also a config file that can be used to change the
configurations of the logger and also the debug mode depending upon 
whether the api is in development or production mode.

tests.py : It contains all the unit tests of the api which could 
be used to make sure that api is working fine. Although if we make changes in the api then sometimes we have to make changes in the unit tests accordingly.

uploads : It is the uploaded image destination where images are saved and removed after sending back the required response.

image_to_bytes: It is the program that converts image into a byte string which can be used to test the api through postman.

crop_f : This is the folder where the cropped images if applicable will be stored and removed automatically at the end of the process.

new_img: This is the folder where the image after correct rotation will be saved and sent back as a response. It will also get automatically removed at the end of the process.


Headers and Body: 

content-type = application/json

Sample request:

url= http://127.0.0.1:5002/
response = requests.post(url=url, data={"image":" image byte string ", “type”:”back or front”})
response.content

Sample response:

{“image”:"b'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4p\nLSwzOko+MzZGNywtQFdBRkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09P\nT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wAARCAe4CdgDASIA\nAhEBAxEB/8QAGwAAAwEBAQEBAAAAAAAAAAAAAAECAwQFBgf/xABAEAACAgEDAwMEAgECAwcDAgcA\nAQIRAwQSIQUxQRMiUQYUMmFCcYEjkVKhsRUkMzRicsEWU5LRQ4Il4aLw8bL/xAAZAQEBAQEBAQAA\nAAAAAA” }


