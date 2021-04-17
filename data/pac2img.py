from scapy.all import PcapReader, raw
from PIL import Image

import numpy as np
import os


def split_pix(pixel):
    h = pixel//16 * 16 + 8
    l = pixel%16 * 16 + 8
    return h, l


def pcap2im(pcap_path, prefix, dest_dir):

    pr = PcapReader(pcap_path)

    for i in range(1000):
        pkt = pr.read_packet()
        if not pkt:
            break
        pcap_hex = raw(pkt)
        pixel_split = [split_pix(b) for b in pcap_hex]
        pixel_all = []

        for h, l in pixel_split:
            pixel_all.append(h)
            pixel_all.append(l)

        if len(pixel_all) > 13*13:
            continue

        print(len(pixel_all))
        pix_arr = np.array(pixel_all)
        pix_arr.resize(13 * 13, refcheck=False)
        pix_im = np.reshape(pix_arr, (13, 13))

        pix_im = np.repeat(pix_im, 3, axis=0)
        pix_im = np.repeat(pix_im, 3, axis=1)

        save_path = os.path.join(dest_dir,"{}_{}.jpg".format(prefix, str(i)))

        im = Image.fromarray(pix_im).convert("RGB")
        im.save(save_path)


if __name__ == '__main__':

    pcap_path1 = "h_res_part_00000_20141202032443.pcap"
    pcap_path2 = "normal_packet.pcap"

    pcap2im(pcap_path1, "honey", "./pcap_im/honeypot_im")
    pcap2im(pcap_path2, "normal", "./pcap_im/normal_im")





