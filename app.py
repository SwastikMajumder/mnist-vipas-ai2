import base64
import cv2
import numpy as np
import streamlit as st
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException


def predict_image(input_data):
    input_data = base64.b64encode(input_data.tobytes()).decode('utf-8')
    model_id = "mdl-7c64gcr10cmb0"
    vps_model_client = model.ModelClient()
    response = vps_model_client.predict(model_id=model_id, input_data=input_data)
    return np.frombuffer(base64.b64decode(response))

thin_arr = [[ [0,0,0],[-1,1,-1],[1,1,1] ],\
            [ [-1,0,0],[1,1,0],[-1,1,-1] ],\
            [ [1,-1,0],[1,1,0],[1,-1,0] ],\
            [ [-1,1,-1],[1,1,0],[-1,0,0] ],\
            [ [1,1,1],[-1,1,-1],[0,0,0] ],\
            [ [-1,1,-1],[0,1,1],[0,0,-1] ],\
            [ [0,-1,1],[0,1,1],[0,-1,1] ],\
            [ [0,0,-1],[0,1,1],[-1,1,-1] ]]
curr_image = None
clean_output = None
digit_list = None
digit_image = None
return_value = None
input_image = None
orig_image = None
def walk(inp):
    global curr_image
    global orig_image
    global clean_output
    global digit_list
    global digit_image
    global return_value
    global input_image
    orig_image = inp
    input_image = orig_image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    input_image = np.array(input_image).copy()
    return_value = []
    digit_list = []
    digit_image = []
    curr_image = np.ones(input_image.shape)*255
    curr_image = curr_image.astype(int)
    clean_output = np.zeros(input_image.shape)
    clean_output = clean_output.astype(int)
    
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if input_image[i][j] == 0 and curr_image[i][j] != 0:
                old_curr_image = curr_image.copy()
                result = connected((i,j))
                if len(result) > 0:
                    tmp = old_curr_image-curr_image
                    tmp[tmp < 0] = 0
                    clean_output += tmp
                    digit_list.append(result)
                    return_value.append(result)
    for i in range(len(digit_list)):
        x_min, y_min, x_max, y_max = digit_list[i]
        digit_image.append(clean_output[x_min:x_max+1, y_min:y_max+1])
        pad_val = round(max(x_max - x_min, y_max - y_min)*1.5)
        digit_image[i] = pad(digit_image[i], pad_val, pad_val)
        background = np.full((pad_val, pad_val), 0, dtype=np.uint8)
        for j in range(16):
            digit_image[i] = 255-(thin((255-digit_image[i])/255, np.array(thin_arr[j%8]))*255)
            background = 255-(thin((255-background)/255, np.array(thin_arr[j%8]))*255)
        digit_image[i] = (digit_image[i]/255)*((255-background)/255)*255
        digit_image[i] = cv2.resize(digit_image[i], (28, 28), interpolation = cv2.INTER_CUBIC)
        resize_thres = 100
        digit_image[i][digit_image[i]>resize_thres]=255
        digit_image[i][digit_image[i]<(resize_thres+1)]=0
        neural_network(digit_image[i]/255)
    return return_value
def display(inp):
    global orig_image
    result = walk(inp)
    size = int(len(result)/2)
    copy_image = orig_image.copy()
    for i in range(size):
        x_min, y_min, x_max, y_max = result[i]
        copy_image = cv2.rectangle(copy_image,(y_min, x_min),(y_max, x_max),(0,0,255),2)
        copy_image = cv2.putText(copy_image,str(result[i+size]),(y_min, x_min),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
    return copy_image
def hit_miss(inp, element):
    output = inp.copy()
    for i in range(1, output.shape[0]-1):
        for j in range(1, output.shape[1]-1):
            arr = inp[i-1:i+2,j-1:j+2]
            tmp = element.copy()
            tmp[arr==-1]=-1
            if np.array_equal(tmp, arr):
                output[i][j] = 1
            else:
                output[i][j] = 0
    return output
def thin(inp, element):
    output = inp.copy()
    tmp = output - hit_miss(output, element)
    tmp[tmp<0]=0
    return tmp
def convert_thin(inp):
    output = inp.copy()
    for i in range(128):
        tmp = thin(output, np.array(thin_arr[i%len(thin_arr)]))
        if np.array_equal(output, tmp):
            break
        else:
            output = tmp
    return output
def position_image(im):
    x_sum = 0
    count = 0
    y_sum = 0
    for i in range(28):
        for j in range(28):
            if im[i][j] == 1:
                count += 1
                if i >= 14:
                    x_sum += i-14.5
                else:
                    x_sum += i-13.5
                if j >= 14:
                    y_sum += j-14.5
                else:
                    y_sum += j-13.5
    output = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            if im[i][j] == 1:
                output[i-round(x_sum/count)][j-round(y_sum/count)] = 1
    return np.array(output)
def neural_network(inp):
    global return_value
    output = inp.copy()
    output = convert_thin(output)
    output = position_image(output)
    output = output.reshape((1,784))
    result = predict_image(output)
    best = -1
    index = 0
    for i in range(10):
        if best < result[i]:
            best = result[i]
            index = i
    return_value.append(index)
def pad(inp, new_height, new_width):
    old_height, old_width = inp.shape
    x_center = (new_width-old_width) //2
    y_center = (new_height-old_height) //2
    output = np.full((new_height, new_width), 0, dtype=np.uint8)
    output[y_center:y_center+old_height,x_center:x_center+old_width] = inp
    return output
def gen_valid(node):
    global input_image
    global curr_image
    x,y = node
    if x==0 or y ==0 or x==input_image.shape[0]-1 or y==input_image.shape[1]-1:
        return []
    valid = [ (x-1,y-1), (x+0,y-1), (x+1,y-1),\
              (x-1,y+0), (x+0,y+0), (x+1,y+0),
              (x-1,y+1), (x+0,y+1), (x+1,y+1),]
    gen = []
    for i in range(9):
        if input_image[valid[i][0]][valid[i][1]] == 0 and curr_image[valid[i][0]][valid[i][1]] != 0:
            gen.append(valid[i])
    return gen
def connected_update(val, node):
    global curr_image
    x_min, y_min, x_max, y_max = val
    curr_image[node[0]][node[1]] = 0
    if x_min > node[0]:
        x_min = node[0]
    if y_min > node[1]:
        y_min = node[1]
    if x_max < node[0]:
        x_max = node[0]
    if y_max < node[1]:
        y_max = node[1]
    return [x_min, y_min, x_max, y_max]
def connected(node):
    curr = set()
    curr_2 = set()
    val = [1000, 1000, -1, -1]
    area = 0
    for i in gen_valid(node):
        area += 1
        val = connected_update(val, i)
        curr.add(i)
    while True:
        if curr == set():
            break
        for i in curr:
            for j in gen_valid(i):
                area += 1
                val = connected_update(val, j)
                curr_2.add(j)
        curr = curr_2
        curr_2 = set()
    if area > 100:
        return val
    else:
        return []
    
def main():
    st.title("Image Fetch and Display")

    # Default image URL
    default_url = "https://raw.githubusercontent.com/SwastikMajumder/notebook_1/refs/heads/main/image.jpg"
    
    # Input field for the image URL with a default value
    image_url = st.text_input("Enter the image URL:", value=default_url)

    if image_url:
        try:
            # Fetch the image from the URL
            response = requests.get(image_url)
            response.raise_for_status()  # Check for HTTP errors

            # Convert the response content to a NumPy array
            file_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Process the image with the display function
            output = display(image)

            # Display the original and processed images using OpenCV
            st.image(image, channels="BGR", caption='Original Image', use_column_width=True)
            st.image(output, caption='Processed Image', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
