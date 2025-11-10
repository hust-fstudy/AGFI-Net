# -*- coding: utf-8 -*-
# @Time: 2024/11/22
# @File: read_file.py
# @Author: fwb
import numpy as np
import os
import struct
import scipy.io as sio


class ReadFile:
    def npz_file_reader(self, filepath):
        """
        :param filepath: DVSGesture dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = np.load(filepath)
        x = events['x']
        y = events['y']
        t = events['t']
        p = events['p']
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def npy_file_reader(self, filepath):
        """
        :param filepath: THU dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = np.load(filepath).astype(np.float32)
        x, y, t, p = events.T
        t_diff = t[1] - t[0]
        if t_diff != 0 and t_diff < 1:
            t = t * 1e6  # timestamp in microseconds
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def aedat_file_reader(self, filepath):
        """
        :param filepath: PAF dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        startTime = 0
        sizeX = 346
        sizeY = 260
        x0 = 0
        y0 = 0
        x1 = sizeX
        y1 = sizeY

        polmask = int('800', 16)
        xmask = int('003FF000', 16)
        ymask = int('7FC00000', 16)
        typemask = int('80000000', 16)
        typedvs = int('00', 16)
        xshift = 12
        yshift = 22
        polshift = 11
        x = []
        y = []
        ts = []
        pol = []
        numeventsread = 0

        length = 0
        aerdatafh = open(filepath, 'rb')
        k = 0
        p = 0
        statinfo = os.stat(filepath)
        if length == 0:
            length = statinfo.st_size

        lt = aerdatafh.readline()
        while lt and str(lt)[2] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            continue

        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        while p < length:
            ad, tm = struct.unpack_from('>II', tmp)
            ad = abs(ad)
            if tm >= startTime:
                if (ad & typemask) == typedvs:
                    xo = sizeX - 1 - float((ad & xmask) >> xshift)
                    yo = float((ad & ymask) >> yshift)
                    polo = 1 - float((ad & polmask) >> polshift)
                    if x0 <= xo < x1 and y0 <= yo < y1:
                        x.append(xo)
                        y.append(yo)
                        pol.append(polo)
                        ts.append(tm)
            aerdatafh.seek(p)
            tmp = aerdatafh.read(8)
            p += 8
            numeventsread += 1
        x = np.array(x)
        y = np.array(y)
        t = np.array(ts)
        p = np.array(pol)
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def mat_file_reader(self, filepath):
        """
        :param filepath: SN-KTH dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = sio.loadmat(filepath)
        x = np.squeeze(events['x'])
        y = np.squeeze(events['y'])
        if ('ts' in events) and ('pol' in events):
            t = np.squeeze(events['ts'])
            p = np.squeeze(events['pol'])
        else:
            t = np.squeeze(events['t'])
            p = np.squeeze(events['p'])
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def bully_data_reader(self, filepath):
        """
        :param filepath: Bully10K dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        data = np.load(filepath, allow_pickle=True)
        data = np.hstack(data)
        x = data['x']
        y = data['y']
        t = data['timestamp']
        p = data['polarity']
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def daily_data_reader(self, filepath, max_events=1e6, start_event=0, x_mask=None):
        """
        :param filepath: Daily dataset file specified by 'filename'
        :param max_events: 1e6
        :param start_event: 0
        :param x_mask: None
        :return: All x, y, t, p in the file
        """
        f = open(filepath, 'rb')
        # Skip header lines.
        bof = f.tell()
        line = f.readline().decode('utf-8').strip()
        tok = '#!AER-DAT'
        version = 0

        while line[0] == '#':
            if line.startswith(tok):
                version = int(float(line[len(tok):].strip()))
            bof = f.tell()
            try:
                line = f.readline().decode('utf-8').strip()
            except UnicodeDecodeError:
                break

        match version:
            case 0:
                version = 1
            case 1:
                version = 1
            case 2:
                version = 2
            case _:
                print(f"Unknown AER-DAT file format version {version}")
        # Version.
        num_bytes_per_event = 6
        match version:
            case 1:
                num_bytes_per_event = 6
            case 2:
                num_bytes_per_event = 8

        f.seek(0, os.SEEK_END)
        num_events = (f.tell() - bof - num_bytes_per_event * start_event) // num_bytes_per_event
        if num_events > max_events:
            print(f"clipping to {max_events} events although there are {num_events} events in file")
            num_events = max_events

        # Read data.
        f.seek(bof + num_bytes_per_event * start_event, os.SEEK_SET)
        dtype_event = np.dtype([
            ('addr', '>u4'),
            ('ts', '>u4')
        ])
        all_events = np.fromfile(f, dtype=dtype_event, count=num_events)
        all_addr = all_events['addr'].astype(np.uint32)  # addr are each 2 bytes (uint16) separated by 4 byte timestamps
        all_ts = all_events['ts'].astype(np.uint32)  # ts are 4 bytes (uint32) skipping 2 bytes after each
        f.close()

        retinaSizeX = 128
        x_shift = None
        y_shift = None
        y_mask = None
        pol_mask = None
        if x_mask is None:
            x_mask = int('fE', 16)
            y_mask = int('7f00', 16)
            x_shift = 1
            y_shift = 8
            pol_mask = 1

        addr = abs(all_addr)
        x = retinaSizeX - 1 - ((addr & x_mask) >> x_shift)  # x addresses
        y = ((addr & y_mask) >> y_shift)  # y addresses
        pol = 1 - 2 * (addr & pol_mask)  # 1 for ON, -1 for OFF

        x = np.array(x)
        y = np.array(y)
        t = np.array(all_ts)
        p = np.array(pol)
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]
