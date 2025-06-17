# ffu-counter
# Copyright (C) 2025 Diyar Polat
#
# This file is part of ffu-counter.
#
# ffu-counter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ffu-counter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ffu-counter.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
import argparse
import os

def remove_well_edges(mask, img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
        param1=50, param2=30,
        minRadius=int(h * 0.35), maxRadius=int(h * 0.55)
    )
    if circles is not None:
        x_c, y_c, r_c = np.round(circles[0, 0]).astype(int)
        well_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(well_mask, (x_c, y_c), r_c, 255, -1)
        mask = cv2.bitwise_and(mask, well_mask)
    edge_mask = np.ones_like(mask) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        near_edge = (x < 10 or y < 10 or x + bw > w - 10 or y + bh > h - 10)
        large = area > 0.008 * w * h
        elong = max(bw, bh) / min(bw, bh) > 8
        if (near_edge and large) or elong:
            cv2.fillPoly(edge_mask, [contour], 0)
    mask = cv2.bitwise_and(mask, edge_mask)
    border_mask = np.zeros_like(mask)
    cv2.rectangle(border_mask, (5,5), (w-5, h-5), 255, -1)
    mask = cv2.bitwise_and(mask, border_mask)
    return mask


def conservative_watershed_split(blob, min_area, max_area, orig_area):
    if orig_area < max_area * 2.2:
        return 1, []

    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    threshold = max(3, 0.4 * dist.max())
    _, fg = cv2.threshold(dist, threshold, 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)
    if cv2.countNonZero(fg) < min_area:
        return 1, []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg = cv2.dilate(blob, kernel, iterations=3)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    m3 = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(m3, markers)

    rects = []
    for m in np.unique(markers):
        if m <= 1:
            continue
        part = (markers == m).astype(np.uint8) * 255
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            a = cv2.contourArea(contour)
            if a < min_area * 0.5 or a > max_area * 1.5:
                continue
            x, y, ww, hh = cv2.boundingRect(contour)
            rects.append((x, y, ww, hh))

    return max(1, len(rects)), rects


def create_magenta_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_magenta1, upper_magenta1 = np.array([140,50,50]), np.array([180,255,255])
    lower_magenta2, upper_magenta2 = np.array([0,50,50]),   np.array([15,255,255])
    mask1 = cv2.inRange(hsv, lower_magenta1, upper_magenta1)
    mask2 = cv2.inRange(hsv, lower_magenta2, upper_magenta2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    b,g,r = cv2.split(img)
    condition = (r>80)&(b>80)&(g<100)&((r.astype(int)+b.astype(int))>1.5*g.astype(int))
    rgb_mask = np.zeros_like(b); rgb_mask[condition] = 255
    return cv2.bitwise_or(hsv_mask, rgb_mask)


def count_ffu(image_path, min_area=100, max_area=4000, show=False, use_magenta=True):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)
    mask = create_magenta_mask(img) if use_magenta else cv2.inRange(
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array([140,50,50]), np.array([180,255,255])
    )
    mask = remove_well_edges(mask, img)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)),2)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)),1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    annotated_image = img.copy(); count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = stats[i, :4]
        if area <= max_area:
            aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
            if aspect_ratio > 4.0:
                continue
            blob = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
                if solidity < 0.25 or circularity < 0.25:
                    continue
            count += 1
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            n, rects = conservative_watershed_split((labels == i).astype(np.uint8) * 255, min_area, max_area, area)
            if n > 1:
                for rx, ry, rw, rh in rects:
                    count += 1
                    cv2.rectangle(annotated_image, (x + rx, y + ry), (x + rx + rw, y + ry + rh), (0, 255, 0), 2)
                    cv2.putText(annotated_image, str(count), (x + rx, y + ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                count += 1
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(annotated_image, f"{count}*", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return count, annotated_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Balanced FFU counter')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('--min-area', type=int, default=120, help='Minimum FFU area')
    parser.add_argument('--max-area', type=int, default=3000, help='Maximum FFU area')
    parser.add_argument('--show', action='store_true', help='Display intermediate windows')
    parser.add_argument('--out', help='Path to save annotated output image')
    args = parser.parse_args()

    count, annotated_image = count_ffu(
        args.image,
        min_area=args.min_area,
        max_area=args.max_area,
        show=args.show,
        use_magenta=True
    )

    print(f"Detected FFU foci: {count}")

    if args.out:
        output_path = args.out
        base, extension = os.path.splitext(output_path)
        if extension.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            output_path = base + '.png'
        output_directory = os.path.dirname(output_path)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        if cv2.imwrite(output_path, annotated_image):
            print(f"Saved annotated image to: {output_path}")
        else:
            print(f"Error: could not write image to {output_path}")
