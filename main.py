import cv2
import numpy as np

def display(title, name):
    cv2.imshow(title, name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

AL = ['AL_ハッチング1_照明あり_IOHD0024 - frame at 0m3s.jpg', 'AL_ハッチング1_照明なし_IOHD0027 - frame at 0m3s.jpg', 'AL_ハッチング2_照明あり_IOHD0025 - frame at 0m3s.jpg', 'AL_ハッチング2_照明なし_IOHD0028 - frame at 0m3s.jpg', 'AL_円のみ_照明あり_IOHD0023 - frame at 0m3s.jpg', 'AL_円のみ_照明あり_IOHD0023 - frame at 0m3s.jpg']
SUS = ['SUS_ハッチング1_照明あり_IOHD0017 - frame at 0m3s.jpg', 'SUS_ハッチング1_照明なし_IOHD0020 - frame at 0m3s.jpg', 'SUS_ハッチング2_照明あり_IOHD0018 - frame at 0m3s.jpg', 'SUS_ハッチング2_照明なし_IOHD0021 - frame at 0m3s.jpg', 'SUS_円のみ_照明あり_IOHD0016 - frame at 0m3s.jpg', 'SUS_円のみ_照明なし_IOHD0019 - frame at 0m3s.jpg']

#img = './AL/' + AL[4]
img = './SUS/' + SUS[0]

img_color = cv2.imread(img)
print(img_color.shape)
display('original', img_color)


img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# ノイズの除去、中央値を採用
img_blur = cv2.medianBlur(img_gray, 9)
display('blur', img_blur)

img_binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 1.5)
display('binary', img_binary)

circles = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=100, param2=20, minRadius=60, maxRadius=120)
# dp=0.8-1.2(解像度=小さいほど厳), minDist=円同士の最低距離, param1=100前後(Canny法の上限閾値)低いほど後検出,
# param2=円検出の閾値(低いと誤検出)
# print(circles)

circles = np.uint16(np.around(circles))# cv2の円描画用にキャスト

if circles is not None:
    img_binary3 = cv2.cvtColor(img_binary,cv2.COLOR_GRAY2RGB)
    for circle in circles[0, :]:
        # 円周を描画する
        cv2.circle(img_binary3, (circle[0], circle[1]), circle[2], (0, 165, 255), 5)
        # 中心点を描画する
        cv2.circle(img_binary3, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    display('after', img_binary3)
        
    for circle in circles[0, :]:
        # 円周を描画する
        cv2.circle(img_color, (circle[0], circle[1]), circle[2], (0, 165, 255), 5)
        # 中心点を描画する
        cv2.circle(img_color, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    display('after', img_color)

cv2.imwrite("after.png", img_color)
