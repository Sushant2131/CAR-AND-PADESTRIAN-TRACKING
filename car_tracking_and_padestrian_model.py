
import cv2
# image:
img_file = ("E:\image.jpg")
# video
video = cv2.VideoCapture("E:\ss\How to Ride a Motorcycle in Market Place [Hindi].mp4")

# pre-trained classifier:
classifier_file = ('car_detector.xml')
# for padestrian
padestrian_file = ('padestrian.xml')


# create car classifier:
car_tracker = cv2.CascadeClassifier(classifier_file)
# padestrian classifier:
padestrian_tracker = cv2.CascadeClassifier(padestrian_file)

# this while loop for prgramme run forever untile car stops
while True:
    # read current frame
    read_successful, frame = video.read()

    if read_successful:
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect car(car is small or big any car can detect):
    cars = car_tracker.detectMultiScale(greyscaled_frame)
    # padestrian detect:
    padestrian = padestrian_tracker.detectMultiScale(greyscaled_frame)
    
    # to prints tha car location hight width
    # print(padestrian)
    # draw a rectangular around the car
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # draw a rectangualar around padestrian:
    for(x, y, w, h) in padestrian:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
   # display image:
    im = cv2.imshow('SELF DRIVING CAR', frame)
    # dont autoclose:
    key = cv2.waitKey(1)
    # stop if Q key pressed:
    if key == 81 or key == 113:
        break


'''
# create opencv image:
img = cv2.imread(img_file)
# convert greysacle:
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create car classifier:
car_tracker = cv2.CascadeClassifier(classifier_file)
# detect car(car is small or big any car can detect):
cars = car_tracker.detectMultiScale(black_n_white)
# draw a rectangular around the car
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 220), 2)


# display image:
im = cv2.imshow('ss car tracking model', img)
# dont autoclose:
cv2.waitKey()
'''
