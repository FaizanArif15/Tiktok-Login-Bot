# from selenium import webdriver
import undetected_chromedriver as webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time



import requests
from bs4 import BeautifulSoup
import os

import cv2
import numpy as np

class Tiktok(webdriver.Chrome):
    
    def __init__(self):
        options = webdriver.ChromeOptions()
        # options.add_experimental_option('detach', True)
        options.add_argument("--headless")
        options.add_argument("--use_subprocess")
        super(Tiktok, self).__init__(options=options)
        self.implicitly_wait(20)
        self.maximize_window()
        
    
    # def open_first_page(self, url='https://www.tiktok.com/en/'):
    def open_first_page(self, url='https://www.tiktok.com/login/phone-or-email/email'):
        self.get(url)
    
    def login_page(self, email, password):
        # Click on this 'Use phone / email / username' button 
        # self.find_element(By.XPATH, '//*[@id="loginContainer"]/div/div/div[1]/div[2]/div[2]/div[2]/div').click()

        # Click on this 'Log in with email or username' button 
        # self.find_element(By.XPATH, '//*[@id="loginContainer"]/div/form/div[1]/a').click()

        # email_element = self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[2]/form/div[1]/input')
        email_element = self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[1]/form/div[1]/input')
        # password_element = self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[2]/form/div[2]/div/input')
        password_element = self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[1]/form/div[2]/div/input')

        email_element.send_keys(email)
        password_element.send_keys(password)

        # self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[2]/form/button').click()
        self.find_element(By.XPATH, '//*[@id="loginContainer"]/div[1]/form/button').click()
            
    def __download_images(self, urls, output_dir):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            if os.path.exists("downloaded_images/image_0.png"):
                os.remove("downloaded_images/image_0.png")
            if os.path.exists("downloaded_images/image_1.png"):
                os.remove("downloaded_images/image_1.png")
    
        for i, url in enumerate(urls):
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(output_dir, f'image_{i}.png'), 'wb') as f:
                    f.write(response.content)
                    # print(f'Downloaded image_{i}.jpg')
    
    
    def download_image(self):
        
        # Find image elements using Selenium
        image_element1 = self.find_element(By.XPATH, '//*[@id="captcha_container"]/div/div[2]/img[1]')
        image_element2 = self.find_element(By.XPATH, '//*[@id="captcha_container"]/div/div[2]/img[2]')
        # Extract image URLs
        image_urls = [image_element1.get_attribute('src'), image_element2.get_attribute('src')]
        # Download the images
        self.__download_images(image_urls, 'downloaded_images')
        
    def __preprocess_image(self, image_path):
            # Load the image and preprocess it
            captcha_image = cv2.imread(image_path)
            gray = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            return blurred
        
    def __find_rotation_angle(self, background_image, rotated_image):
            # Initialize ORB detector
            orb = cv2.ORB_create()

            # Find keypoints and descriptors
            keypoints1, descriptors1 = orb.detectAndCompute(background_image, None)
            keypoints2, descriptors2 = orb.detectAndCompute(rotated_image, None)

            # Initialize brute-force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            matches = bf.match(descriptors1, descriptors2)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Take the best match
            best_match = matches[0]

            # Get the keypoints for the best match
            kp1 = keypoints1[best_match.queryIdx]
            kp2 = keypoints2[best_match.trainIdx]

            # Calculate the rotation angle
            # angle_radians = np.arctan2(kp2.pt[1] - kp1.pt[1], kp2.pt[0] - kp1.pt[0])
            # angle_degrees = np.degrees(angle_radians)
            angle_degrees = int(kp2.pt[0] - kp1.pt[0])
            return angle_degrees
            
            
            
        
    
    def __rotate_captcha(self, slider_element, rotation_angle):
            # Perform actions to move the slider button according to the rotation angle
            action = ActionChains(self)
            action.click_and_hold(slider_element).perform()
            action.move_by_offset(rotation_angle, 0).perform()
            action.release().perform()
            time.sleep(2)
    
    def rotate_image(self, image, homography_matrix):
        # Apply homography matrix to rotate the image
        if homography_matrix is not None:
            rotated_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
            return rotated_image
        else:
            return image
    
    def calculate_distance_to_slider(self, rotated_image, background_image, homography_matrix):
        # Calculate the distance needed to move the slider button
        # Calculate the centroids of both images
        centroid_rotated = np.array([rotated_image.shape[1] // 2, rotated_image.shape[0] // 2])

        # Calculate the centroid of the rotated image in the background image's coordinates
        if homography_matrix is not None:
            transformed_centroid = np.dot(homography_matrix, np.array([centroid_rotated[0], centroid_rotated[1], 1]))
            centroid_background = np.array([transformed_centroid[0] / transformed_centroid[2], transformed_centroid[1] / transformed_centroid[2]])

            # Calculate the distance between centroids
            distance = np.linalg.norm(centroid_background - centroid_rotated)
        else:
            centroid_rotated = np.array([rotated_image.shape[1] // 2, rotated_image.shape[0] // 2])
            centroid_background = np.array([background_image.shape[1] // 2, background_image.shape[0] // 2])
            
            # Calculate the distance between centroids
            distance = np.linalg.norm(centroid_background - centroid_rotated)
            
        
        return distance
    
    def find_homography_matrix(self, background_image, rotated_image):        
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(background_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(rotated_image, None)

        # Initialize brute-force matcher
        bf = cv2.BFMatcher()

        # Match descriptors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)

        # Compute homography if enough good matches are found
        print(len(good_matches))
        if len(good_matches) > 10:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            return H
        else:
            return None
           
    def solve_captcha(self):
        
        # Wait for the CAPTCHA widget to load 
        try:
            pass
            slider = WebDriverWait(self, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="captcha_container"]/div/div[3]/div[1]')))
            # print(slider.text) //*[@id="captcha_container"]/div/div[3]/div[1]
        except:
            print("Slider not found.")
            self.quit()
        # Get the slider button
        slider_button = slider.find_element(By.XPATH, '//*[@id="secsdk-captcha-drag-wrapper"]/div[2]')

        
        # Load the background image and the rotated captcha image
        background_image = self.__preprocess_image('downloaded_images/image_0.png')
        rotated_captcha = self.__preprocess_image('downloaded_images/image_1.png')

        
        
        # Find the homography matrix to align the rotated captcha image with the background image
        homography_matrix = self.find_homography_matrix(background_image, rotated_captcha)

        # Rotate the captcha image using the homography matrix
        corrected_captcha = self.rotate_image(rotated_captcha, homography_matrix)

        # Calculate the distance needed to move the slider button
        distance_to_slider = self.calculate_distance_to_slider(corrected_captcha, background_image, homography_matrix)
        print("Distance to slider:", distance_to_slider)
        self.__rotate_captcha(slider_button, distance_to_slider)
        print('move slider')
        
        # # Find the rotation angle needed to align the captcha image with the background image
        # rotation_angle = self.__find_rotation_angle(background_image, rotated_captcha)

        
        # Rotate the captcha image using Selenium to match the background image
        # self.__rotate_captcha(slider_button, rotation_angle)
        

        # Wait for some time to verify the captcha solution
        time.sleep(5)

        # Close the browser
        # driver.quit()

    
if __name__ == "__main__":
    instance = Tiktok()
    instance.open_first_page()
    email = input("Please enter email")
    password = input("Please enter Password")
    instance.login_page(email=email, password=password)
    time.sleep(60)
    # print(instance.title)
    # while instance.title == 'Log in | TikTok':
    #     print(instance.title)
    #     # time.sleep(3)
    #     instance.download_image()
        # instance.solve_captcha()
    # else:
    #     print(instance.title)
    #     print('Sucessfully pass captcha')
    
    

