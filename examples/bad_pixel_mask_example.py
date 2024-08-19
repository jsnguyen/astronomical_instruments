from astronomical_instruments import NIRC2

def main():
    bad_pixel_mask = NIRC2.make_bad_pixel_mask_20230101(save_mask=True)
        
if __name__ == '__main__':
    main()