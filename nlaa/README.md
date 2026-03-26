
原始的可运行的工程路径为：/data2/whd/workspace/NAAN/NAAN

pip install tensorflow-gpu==1.9.0 (compatible with CUDA10.1 and cudnn 7.5)
pip install h5py
easy_install pillow


Install:
https://blog.csdn.net/sinat_33486980/article/details/95078922

CUDNN problem:
https://blog.csdn.net/wukai0909/article/details/97489794



{Error using instrument
The instrument object requires JAVA support.

Error in icinterface (line 27)
            obj = obj@instrument(validname);

Error in tcpip (line 71)
            obj = obj@icinterface('tcpip'); %#ok<PROP>
}
{Undefined function or variable 'client_tcp'.
}
{Error using fwrite
Invalid file identifier. Use fopen to generate a valid file identifier.

Error in track_frame (line 109)
            fwrite(client_tcp, 'client ok');

Error in MOT_associate (line 9)
    tracker = track_frame(tracker, fr, frame_image, bboxes_associate,
    index_det, seq_name, opt);

Error in track_seq (line 150)
                trackers{ind} = MOT_associate(fr, frame_image, frame_size,
                bboxes_associate, trackers{ind}, opt, seq_name);
}
{Undefined function or variable 'test_time'.
}
{^HError using fclose
Invalid file identifier. Use fopen to generate a valid file identifier.
}^



