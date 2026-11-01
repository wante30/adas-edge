"""Keep both processes on the same shared-frame/source contract."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('test_session',Path(__file__).parents[1]/'tools/test_session.py')
session=importlib.util.module_from_spec(spec);spec.loader.exec_module(session)

class SessionContract(unittest.TestCase):
    def test_local_replay_preserves_shared_identity_and_normal_commands(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);evidence=root/'evidence';evidence.mkdir()
            build=root/'build-stage4';build.mkdir();(build/'qt_gpu_probe').touch()
            boot=root/'boot';boot.write_text('test-boot',encoding='utf-8')
            state=evidence/'session.json'
            state.write_text(json.dumps({'boot_id':'test-boot','baseline_hmi':{
                'args':['old_hmi','--pgv-file','old.pgv','--source-label','pgv-live',
                        '--ai-frame-file','old.mfr1','--ai-hz','10'],'env':{}}}),encoding='utf-8')
            calls=[]
            def launch(args,cwd,env,log):
                calls.append((args,env));return len(calls)+100
            args=types.SimpleNamespace(build='build-stage4',label='contract',raw_replay='long.rgb',
                                       ai_hz='0',action_port='38262',external_qml=True)
            with patch.multiple(session,ROOT=root,EVIDENCE=evidence,STATE=state,BOOT=boot,launch=launch,stop=lambda _:None):
                with contextlib.redirect_stdout(io.StringIO()):session.start(args)
            ai,env=calls[0];hmi,hmi_env=calls[1]
            self.assertEqual(ai[ai.index('--live-frame-file')+1],hmi[hmi.index('--ai-frame-file')+1])
            self.assertEqual(ai[ai.index('--source')+1],'LOCAL_REPLAY')
            self.assertNotIn('--pgv-file',hmi)
            self.assertEqual(hmi[hmi.index('--raw-replay')+1],'long.rgb')
            self.assertEqual(hmi[hmi.index('--ai-hz')+1],'0')
            self.assertEqual(ai[ai.index('--ocr-hz')+1],'2')
            self.assertEqual(env['ADAS_TEST_ACTION_PORT'],'38262')
            self.assertIn('QT_QML_FILE',hmi_env)

if __name__=='__main__':unittest.main()
