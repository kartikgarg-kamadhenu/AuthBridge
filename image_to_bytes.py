import base64


# creating a dummy byte string to test the api using postman
with open("000016.jpg","rb") as f:
    image_to_bytes=base64.encodebytes(f.read())
    
    
# we will copy this base64 byte string    
print(image_to_bytes)
