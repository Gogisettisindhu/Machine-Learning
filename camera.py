import cv2
#pip install opencv-python
'''img=cv2.imread('a.jpeg')
print(img)
cv2.imwrite('copy.jpeg',img)
cv2.imshow('vignan',img)
cv2.waitKey(0)
'''

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('vigna',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cv2.destroyAllWindows()