# modified from : https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment/blob/main/app.py

import streamlit as st
import torch
import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

def get_accuracy_str(raw_list): # display detection result as string 

    label_found = []

    for each_label in raw_list:
        label_found.append([each_label[-1], round((each_label[-3]*100),2) ])

    label = []

    for i in range(len(label_found)):
        label.append(label_found[i][0])

    label = list(set(label)) # save a only 1 found label 


    current_label = ""

    count = 0
    
    sum_result = []

    for j in range(len(label_found)): # get only 1 labels and most accuracy in result_list
        for k in range(len(label)):
            #print(label_found[j], label[k])
            if (label_found[j][0] == label[k]):
                if current_label != label[k]:
                    current_label = label[k]
                    count += 1
                    if count <= len(label):
                        sum_result.append(label_found[j])#, label[k])

    for each_label in sum_result: # print detection result

        # change to thainame and add treatment and recommend result
        if each_label[0] == "Powdery Mildew":
            label_name = "โรคราแป้ง"
            recommend = "None"
            treatment = "-	หลังจากเก็บเกี่ยวผลผลิตแล้ว ให้ทำลายเศษซากพืชที่เคยเป็นโรคโดยไถกลบ และปลูกพืชหมุนเวียน \n -	ฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น ไตรอะดิมีฟอน (triadimefon) ไมโคลบิวทานิล (myclobutanil) โพรพิโคนาโซล (propiconazole) อะซอกชีสโตรบิน (azoxystrobin)"
        elif each_label[0] == "Spot":
            label_name = "โรคใบจุด"
            recommend = "-	ตัดแต่งกิ่งทุเรียนให้โปร่ง แสงแดดส่องได้ทั่วถึง \n  - เมื่อพบกิ่ง และใบเริ่มแสดงอาการของโรคเพียงเล็กน้อย ให้ตัด และรวบรวมเผาทำลาย รวมทั้งให้รวบรวมใบที่ร่วงหล่นอยู่เผาทิ้งด้วยเพื่อลดการสะสมเชื้อโรค และลดการระบาดในปีต่อไป "
            treatment = "-	ฉีดพ่นด้วยสารป้องกันกำจัดเชื้อรา เช่น โพรคลอราซ (prochloraz) แมนโคเซบ (mancozeb) ไดฟีโนโคนาโซล (difenoconazole) เป็นต้น"
        elif each_label[0] == "Blight":
            label_name = "โรคใบไหม้"
            recommend = "-	ตัดแต่งใบที่เป็นโรค รวมทั้งกำจัดวัชพืชบริเวณแปลงปลูก เพื่อลดแหล่งสะสมของเชื้อสาเหตุ"
            treatment = "-	ตัดกิ่งที่เป็นโรคออกเผาทำลาย (ถ้าเป็นกิ่งใหญ่ควรทาลอยปูนแดงหรือสารประกอบทองแดง) แล้วให้ฉีดพ่นด้วยสารเคมีคาร์เบ็นดาชิม (carbendazim) 60 % WP  อัตรา 10 กรัมต่อน้ำ 20 ลิตร หรือสารเคมีคอปเปอร์ออกซีคลอไรด์ (copper oxychloride) 85 % WP อัตรา 50 กรัมต่อน้ำ 20 ลิตร ให้ทั่วทั้งภายใน และภายนอก"
        elif each_label[0] == "N_loss":
            label_name = "อาการขาดธาตุไนโตรเจน"
            recommend = "None"
            treatment = "-	ผสมปุ๋ยทางดิน : ผสมปุ๋ย NPK ที่มีอัตราส่วนของค่า N มากที่สุด และสังเกตปริมาณการใช้ตามอาการของใบ \n -	ผสมปุ๋ยทางใบ : ใช้ปุ๋ยเคมีที่มีค่า N สูงๆ หรือ ใช้ยูเรียน้ำ สูตรไนโตรเจนสูง ผสมแล้วพ้นปุ๋ยนํ้า"

        st.success(f'มีโอกาสเป็น :    "{label_name}"     {each_label[1]} % ')
        if recommend != "None":
            st.write(f"คำแนะนำในการรักษา : {label_name}")
            st.write(recommend)
        st.write(f"วิธีการรักษา : {label_name}")
        st.write(treatment)




def imageInput(device):
    image_file = st.file_uploader(label = "Upload Durian leaf here.. (ใส่รูปภาพตรงนี้)",type=['png','jpg','jpeg'])
    if image_file is not None:

        st.write("## Detection result.. (สรุปผลการวิเคราะห์)")

        #img = Image.open(image_file)
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
        outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())

        # call Model prediction--
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/DurAIn-ni_yolo1.pt', force_reload=True)
        _ = model.cuda() if device == 'cuda' else model.cpu() # hide cuda_cnn display source : https://stackoverflow.com/questions/41149781/how-to-prevent-f-write-to-output-the-number-of-characters-written
        pred = model(imgpath)
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)

        pred.save()
        detect_val = (pred.pandas().xyxy[0]).values.tolist()



        # --Display predicton / print result
        img_ = Image.open(outputpath)
        st.image(img_)

        get_accuracy_str(detect_val) # get detection string result

    else:
        st.write("## Waiting for image.. (รอผู้ใช้งานใส่รูปภาพ)")
        st.image('./ania.png')
    
    st.caption("Made by Tanaanan MwM")
            
    
            
            
                

def main(): 
    st.sidebar.image("./durian_logo.png")
    st.sidebar.header("DurAIn-ni Webapp." + "\n" + "ระบบวิเคราะห์อาการป่วยของต้นทุเรียนด้วย AI (YOLOV5_model)")

    st.sidebar.title('⚙️ Select option')
    activities = ["Detection (วิเคราะห์โรค)", "About (เกี่ยวกับ)"]
    choice = st.sidebar.selectbox("Select option.. (เลือกโหมด)",activities)



    #st.sidebar.markdown("https://bit.ly/3uvYQ3R")

    if choice == "Detection (วิเคราะห์โรค)":

        st.header('Detection (วิเคราะห์โรค)')
        st.subheader('⚙️ Select option')

        col1, col2 = st.columns(2)

        with col2:
            # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
            if torch.cuda.is_available():
                deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=1)
            else:
                deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=0)
            # -- End of Sidebar
        with col1:
            pages_name = ['Upload own data', 'From test set']
            page = st.radio('Select option mode :', pages_name) 

        if page == "Upload own data":
            st.subheader('📸 Upload image')
            t1 = time.perf_counter()
            imageInput(deviceoption)
            t2 = time.perf_counter()
            st.write('time taken to run: {:.2f} sec'.format(t2-t1))

        elif page == "From test set":
            st.warning("On the future progress... ")

    
    elif choice == 'About (เกี่ยวกับ)' :
        st.header("About... (เกี่ยวกับ)")
        st.subheader("DurAIn-ni คืออะไร ?")
        st.write("- เป็นระบบที่วิเคราะห์โรคที่เกิดได้จากต้นทุเรียนผ่านใบ พร้อม แนะนำวิธีในการรักษาเบื้องต้น โดยใช้ ระบบ AI")

        st.subheader("DurAIn-ni ทำอะไรได้บ้าง ?")
        st.write("- วิเคราะห์โรคจากต้นทึเรียนโดยใช้ใบ (Object detection)")
        st.write("- ให้คำแนะนำในการรักษาเบื้องต้น")
        st.write("- โดยในปัจจุบันสามารถวิเคราะหได้ 5 โรค หลักๆ ได้แก่ " + "\n" + "  1. โรคราแป้ง (Powdery mildew disease)" + "\n" + "  2. โรคใบไหม้ (Leaf blight disease)" + "\n" + "  3. โรคใบจุด (Leaf spot disease)" + "\n" + "  4. อาการขาดธาตุ แมกนีเซียม (Magnesium deficiency)" + "\n" + "  5. อาการขาดธาตุ ไนโตรเจน (Nitrogen deficiency)")

        st.subheader("ทำไมถึงต้องเลือกใช้ DurAIn-ni ")
        st.write("- สามารถใช้วินิจฉัยโรคที่เกิดจากต้นทุเรียนเบื้องต้นได้ด้วยตัวเองในทันที โดยไม่จำเป็นที่จะต้องมีผู้เชี่ยวชาญในการวิเคราะห์ตลอดเวลา พร้อมบอกรายละเอียดโรค และ วิธีการรักษาในเบื้องต้น ดังนั้น  เกษตรกร และ คนทั่วไป สามารถวิเคราห์โรคที่เกิดจากต้นทุเรียน และ สามารถที่จะรักษาได้ได้โดยทันท่วงที ")



        st.subheader("คำแนะนำในการใช้งาน")
        st.write("- ภาพไม่ควรมีแสงที่สว่างมากเกืนไป และ มืดเกินไป มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อนเอาได้")#4
        st.write("- ภาพไม่ควรที่จะอยู่ไกลเกินไป และ ควรมีความชัด มิฉะนั้นอาจทำให้การตรวจจับคลาดเคลื่อน หรือ ไม่สามารถตรวจจับได้")#5

        st.subheader("รายละเอียดเพิ่มเติม")
        st.write('[โรคพื้นฐานที่เกิดในต้นทุเรียน](https://kasetgo.com/t/topic/401483)')
        st.write('[สถิติการส่งออกทุเรียนของไทย](http://impexp.oae.go.th/service/export.php?S_YEAR=2560&E_YEAR=2565&PRODUCT_GROUP=5252&PRODUCT_ID=4977&wf_search=&WF_SEARCH=Y)')

    st.sidebar.subheader('Made by Tanaanan MwM')
    st.sidebar.write("Contact : mjsalyjoh@gmail.com")

if __name__ == '__main__':
    main()

