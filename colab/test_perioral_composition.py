"""Isolated synthetic CPU tests; fixtures are not real-model/visual approval."""
import copy
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker


def face68(angle=0):
    """Artificial anatomical test data ONLY; production never synthesizes it."""
    points = [[120.0, 120.0, .95] for _ in range(68)]
    for index in range(17):
        theta = math.pi - index * math.pi / 16
        points[index] = [120 + 70 * math.cos(theta), 110 + 80 * math.sin(theta), .95]
    for index in range(17, 27):
        points[index] = [85 + (index - 17) * 7, 85, .95]
    for index in range(27, 31):
        points[index] = [120, 93 + (index - 27) * 10, .95]
    for index, value in enumerate([(108,127),(114,131),(120,133),(126,131),(132,127)], 31):
        points[index] = [*value, .95]
    for start, center in ((36,100),(42,140)):
        for index in range(6):
            theta = math.pi + index * math.pi / 3
            points[start+index] = [center + 10*math.cos(theta), 100+5*math.sin(theta), .95]
    for start, count, rx, ry in ((48,12,20,10),(60,8,12,5)):
        for index in range(count):
            theta = math.pi + index*2*math.pi/count
            points[start+index] = [120+rx*math.cos(theta),150+ry*math.sin(theta),.95]
    rotation = math.radians(angle)
    for point in points:
        x, y = point[0]-120, point[1]-150
        point[0] = 120+x*math.cos(rotation)-y*math.sin(rotation)
        point[1] = 150+x*math.sin(rotation)+y*math.cos(rotation)
    return points


def evidence(source, boxes, start=0, points=None):
    binding = worker.perioral_landmark_binding(source, boxes, 240, 240, start)
    return {**binding, 'device':'cpu', 'model_sha256':'1'*64,
            'frames':[{'frame_index':binding['start_frame']+index,
                       'points':copy.deepcopy(points or face68())} for index in range(len(boxes))]}


class PerioralGeometryTests(unittest.TestCase):
    box = (20,20,220,220)

    def test_eyes_brows_forehead_zero_and_mouth_changes_at_rotated_poses(self):
        for angle in (-45,-25,0,25,45):
            with self.subTest(angle=angle):
                points = face68(angle)
                mask = worker.perioral_mask_frame(240,240,self.box,points)
                self.assertEqual(len(mask),240*240)
                for index in [*range(17,27),*range(36,48)]:
                    x,y,_ = points[index]
                    self.assertEqual(mask[int(y)*240+int(x)],0)
                self.assertEqual(mask[150*240+120],255)
                for y in range(240):
                    for x in range(240):
                        if not (20<=x<220 and 20<=y<220):
                            self.assertEqual(mask[y*240+x],0)

    def test_feather_is_inside_polygon_not_a_blur_that_expands_it(self):
        polygon,_ = worker.perioral_landmark_polygon(240,240,self.box,face68())
        mask = worker.perioral_mask_frame(240,240,self.box,face68())
        self.assertTrue(any(0<value<255 for value in mask))
        self.assertEqual(mask[50*240+120],0)
        self.assertEqual(mask[110*240+120],0)
        for y in range(240):
            for x in range(240):
                if mask[y*240+x]:
                    self.assertTrue(all((b[0]-a[0])*(y+.5-a[1])-(b[1]-a[1])*(x+.5-a[0])>=-1e-7
                                        for a,b in zip(polygon,polygon[1:]+polygon[:1])))

    def test_absent_low_confidence_nonfinite_and_unowned_landmarks_fail_closed(self):
        bad = [None, [], [(1,2,.9)]*67]
        for index, part, value in ((48,2,.54),(30,0,float('nan')),(36,0,-1),(54,0,True),(8,2,1.2)):
            points=face68();points[index][part]=value;bad.append(points)
        for points in bad:
            with self.subTest(points=str(points)[:50]),self.assertRaises(worker.WorkerError):
                worker.perioral_mask_frame(240,240,self.box,points)

    def test_inconsistent_nose_eye_mouth_anatomy_fails_instead_of_using_face_box(self):
        for index,position in ((30,(120,155)),(8,(120,100)),(36,(100,145))):
            points=face68();points[index][:2]=position
            with self.subTest(index=index),self.assertRaises(worker.WorkerError):
                worker.perioral_mask_frame(240,240,self.box,points)


class PerioralBindingAndGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='.test-perioral-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'source.bin';self.source.write_bytes(b'synthetic source binding')
        self.boxes=[(20,20,220,220)]*2
        self.config=self.root/'musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        self.config.parent.mkdir(parents=True);self.config.write_text('# synthetic config')
        self.weight=self.root/'models/dwpose/dw-ll_ucoco_384.pth'
        self.weight.parent.mkdir(parents=True);self.weight.write_bytes(b'synthetic weights')
        self.environment={'DUBSTUDIO_MUSETALK_ROOT':str(self.root),'DUBSTUDIO_MUSETALK_PYTHON':sys.executable}

    def test_old_contract_other_source_boxes_clock_model_or_count_are_rejected(self):
        valid=evidence(self.source,self.boxes)
        binding=worker.perioral_landmark_binding(self.source,self.boxes,240,240,0)
        self.assertEqual(len(worker.validate_perioral_landmarks(valid,binding,self.boxes)),2)
        changes=[('contract','tracked-face-box-only-inner-feather-v1'),('source_sha256','2'*64),
                 ('boxes_sha256','3'*64),('fps',30),('musetalk_commit','old'),('model_sha256','not-a-hash'),
                 ('device','auto'),('frames',valid['frames'][:1])]
        for key,value in changes:
            invalid=copy.deepcopy(valid);invalid[key]=value
            with self.subTest(key=key),self.assertRaises(worker.WorkerError):
                worker.validate_perioral_landmarks(invalid,binding,self.boxes)
        invalid=copy.deepcopy(valid);invalid['frames'][1]['frame_index']=25
        with self.assertRaises(worker.WorkerError):worker.validate_perioral_landmarks(invalid,binding,self.boxes)

    def test_low_confidence_later_frame_blocks_before_mask_encoder(self):
        invalid=evidence(self.source,self.boxes);invalid['frames'][1]['points'][48][2]=.2
        with patch.object(worker.subprocess,'Popen') as encoder,self.assertRaisesRegex(worker.WorkerError,'quadro 1'):
            worker.write_spatial_face_mask(self.root/'mask.mkv',self.boxes,240,240,25,self.root/'log',
                                          reference_video=self.source,landmark_payload=invalid)
        encoder.assert_not_called()

    def test_missing_original_never_writes_full_face_mask(self):
        with patch.object(worker.subprocess,'Popen') as encoder,self.assertRaisesRegex(worker.WorkerError,'rosto inteiro'):
            worker.write_spatial_face_mask(self.root/'mask.mkv',self.boxes,240,240,25,self.root/'log')
        encoder.assert_not_called()

    def test_noncanonical_start_cannot_relabel_landmarks(self):
        for start in (.01,True,float('nan'),'0'):
            with self.subTest(start=start),self.assertRaises(worker.WorkerError):
                worker.perioral_landmark_binding(self.source,self.boxes,240,240,start)

    def test_failed_mask_rasterization_terminates_encoder_without_whole_face_fallback(self):
        with patch.object(worker.subprocess,'Popen') as encoder,\
                patch.object(worker,'perioral_mask_frame',side_effect=worker.WorkerError('synthetic raster failure')):
            with self.assertRaisesRegex(worker.WorkerError,'synthetic raster failure'):
                worker.write_spatial_face_mask(self.root/'mask.mkv',self.boxes,240,240,25,self.root/'log',
                                              reference_video=self.source,landmark_payload=evidence(self.source,self.boxes))
        encoder.return_value.kill.assert_called_once();encoder.return_value.wait.assert_called_once()
        self.assertFalse((self.root/'mask.perioral.json').exists())

    def test_missing_current_mask_proof_blocks_composition_before_ffmpeg(self):
        metadata={'streams':[{'codec_type':'video','width':240,'height':240,'r_frame_rate':'25/1','avg_frame_rate':'25/1'}]}
        with patch.object(worker,'ffprobe',return_value=metadata),\
                patch.object(worker,'video_stream_duration',return_value=1),\
                patch.object(worker,'video_frame_count',return_value=25),\
                patch.object(worker,'write_spatial_face_mask'),patch.object(worker,'run') as composition:
            with self.assertRaisesRegex(worker.WorkerError,'Falta a prova'):
                worker.compose_face_patch(self.source,self.source,self.source,0,.08,1,self.root,self.root/'log',
                                          self.boxes,25,(240,240))
        composition.assert_not_called()

    def fake_run(self,calls,device='cpu',mutate_source=False):
        def execute(command,**options):
            calls.append((command,options))
            if command[0]=='git':return subprocess.CompletedProcess(command,0,worker.PERIORAL_MUSE_COMMIT+'\n','')
            request=json.loads(Path(command[-2]).read_text())
            self.assertEqual(request['device'],device)
            self.assertEqual(options['env']['MPLBACKEND'],'Agg')
            if device=='cpu':self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'],'')
            script=command[2]
            self.assertNotIn('from musetalk.utils.preprocessing',script)
            self.assertIn('device=device',script);self.assertIn('inference_topdown(model, crop',script)
            self.assertIn('face = points[0,23:91]',script);self.assertIn('instances.keypoint_scores',script)
            compile(script,'perioral-subprocess','exec')
            payload={key:value for key,value in request.items() if key not in {'source','boxes'}}
            payload['frames']=evidence(self.source,self.boxes)['frames']
            Path(command[-1]).write_text(json.dumps(payload))
            if mutate_source:self.source.write_bytes(b'source changed during inference')
            return subprocess.CompletedProcess(command,0,'synthetic landmark completion','')
        return execute

    def test_cpu_is_explicit_model_is_loaded_once_and_valid_cache_does_not_rerun(self):
        calls=[]
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',side_effect=self.fake_run(calls)):
            first=worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log',device='cpu')
            second=worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log',device='cpu')
        self.assertEqual(first,second);self.assertEqual(sum(command[0]==sys.executable for command,_ in calls),1)

    def test_default_gpu_is_explicit_not_automatic_cpu_fallback(self):
        calls=[]
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',side_effect=self.fake_run(calls,device='cuda:0')):
            payload=worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log')
        self.assertEqual(payload['device'],'cuda:0')

    def test_changed_source_during_inference_never_promotes_landmark_cache(self):
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',side_effect=self.fake_run([],mutate_source=True)):
            with self.assertRaisesRegex(worker.WorkerError,'mudaram'):
                worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log',device='cpu')
        self.assertFalse(list(self.root.glob('perioral-landmarks-*.json')))
        self.assertFalse(list(self.root.glob('.perioral-*.json')))

    def test_unreadable_current_landmark_cache_fails_with_clear_error(self):
        calls=[]
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',side_effect=self.fake_run(calls)):
            worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log',device='cpu')
            cached=next(self.root.glob('perioral-landmarks-*.json'));cached.write_text('{broken json')
            with self.assertRaisesRegex(worker.WorkerError,'cache.*ilegível'):
                worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log',device='cpu')
        self.assertEqual(sum(command[0]==sys.executable for command,_ in calls),1)

    def test_requested_gpu_failure_is_not_retried_on_cpu(self):
        calls=[]
        def failed(command,**options):
            calls.append(command)
            if command[0]=='git':return subprocess.CompletedProcess(command,0,worker.PERIORAL_MUSE_COMMIT,'')
            self.assertIn('not torch.cuda.is_available()',command[2])
            self.assertIn('não haverá fallback CPU',command[2])
            return subprocess.CompletedProcess(command,1,'','GPU unavailable (synthetic)')
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',side_effect=failed):
            with self.assertRaisesRegex(worker.WorkerError,'GPU unavailable'):
                worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log')
        self.assertEqual(len(calls),2);self.assertFalse(list(self.root.glob('perioral-landmarks-*.json')))

    def test_unknown_muse_commit_never_invokes_landmark_model(self):
        with patch.dict(os.environ,self.environment),patch.object(worker.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'unknown','')) as calls:
            with self.assertRaisesRegex(worker.WorkerError,'revisão'):
                worker.generate_perioral_landmarks(self.source,self.boxes,240,240,0,self.root,self.root/'log')
        self.assertEqual(calls.call_count,1)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class RealSyntheticPerioralCompositionTests(unittest.TestCase):
    def test_existing_render_recomposes_only_mouth_and_keeps_original_clock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='.test-perioral-video-') as folder:
            root=Path(folder);source=root/'source.mp4';render=root/'existing-render.mp4'
            for target,color,rate,duration,filter_args in ((source,'blue',30,2,['-vf','drawbox=x=30:y=20:w=180:h=90:color=green:t=fill']),
                                                          (render,'white',25,1,[])):
                subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',f'color={color}:size=240x240:rate={rate}:duration={duration}',
                                *filter_args,'-an','-c:v','libx264','-preset','ultrafast','-crf','16','-pix_fmt','yuv420p',str(target)],check=True,capture_output=True,timeout=30)
            boxes=[(20,20,220,220)]*25;original_hash=worker.sha256(source);render_hash=worker.sha256(render)
            output=worker.compose_face_patch(render,source,source,0,1,1,root,root/'log',boxes,25,(240,240),
                                              landmark_payload=evidence(source,boxes))
            self.assertEqual(worker.video_frame_count(output),worker.video_frame_count(source))
            self.assertEqual(worker.sha256(source),original_hash);self.assertEqual(worker.sha256(render),render_hash)
            def first_frame(path):
                return subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','gray','pipe:1'],check=True,capture_output=True,timeout=30).stdout
            original,final=first_frame(source),first_frame(output)
            self.assertLess(abs(original[50*240+120]-final[50*240+120]),3)
            self.assertLess(abs(original[100*240+100]-final[100*240+100]),3)
            self.assertGreater(final[150*240+120]-original[150*240+120],100)
            proof=json.loads((root/'musetalk-perioral-source-mask-v1-01.perioral.json').read_text())
            self.assertEqual(proof['contract'],worker.PERIORAL_MASK_CONTRACT)
            self.assertTrue(proof['composition_completed']);self.assertEqual(proof['human_visual_review'],'pending')
            self.assertEqual(proof['render_sha256'],render_hash);self.assertEqual(proof['output_sha256'],worker.sha256(output))


if __name__=='__main__':unittest.main()
