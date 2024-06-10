import functions as fc

def main():
    image_path = "sample.jpg"
    save_path = "output.png"

    output = fc.query(image_path)

    areas = fc.calculate_pixel_area(output)
    car_index = []
    road_index = None
    sidewalk_index = None

    for i in range(len(areas)):
        label = areas[i]['label']
        print(f"{i}. Label: {label}, Pixels: {areas[i]['pixels']}")
        if label == 'car':
            car_index.append(i)
        elif label == 'road':
            road_index = i
        elif label == 'sidewalk':
            sidewalk_index = i
    
    n = len(car_index)
    for i in range(n):
        for j in range(0, n-i-1):
            if areas[car_index[j]]['pixels'] < areas[car_index[j+1]]['pixels']:
                car_index[j], car_index[j+1] = car_index[j+1], car_index[j]

    print(car_index)
    print(road_index)
    print(sidewalk_index)

    if road_index != None:
        output_image_1 = fc.overlay_mask_on_image(mask_data = output,
                                                index = road_index,
                                                original_image_path = image_path,
                                                color = (255,0,0))
        output_image_1.save("output1.png")
        
    if sidewalk_index != None:
        output_image_2 = fc.overlay_mask_on_image(mask_data = output,
                                                index = sidewalk_index,
                                                original_image_path = image_path,
                                                color = (0,255,0))
        output_image_2.save("output2.png")

    if car_index != None:
        output_image_3 = fc.overlay_mask_on_image(mask_data = output,
                                                index = car_index[0],
                                                original_image_path = image_path,
                                                color = (0,0,255))
        output_image_3.save("output3.png")

if __name__ == "__main__":
    main()