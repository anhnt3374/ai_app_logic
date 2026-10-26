import base64
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .protocol import send, receive
from ..config import resolve_environment


class WorkerClient:
    def __init__(self,target,timeout=120):
        self.timeout=timeout
        self.process=None
        self.ssh=None
        if target.location=='host':
            environment=os.environ.copy()
            package_root=str(Path(__file__).resolve().parents[2])
            environment['PYTHONPATH']=package_root+os.pathsep+environment.get('PYTHONPATH','')
            if target.runtime=='openvino':
                # OpenVINO's documented CI mode disables SDK telemetry/consent writes.
                environment['CI']='true'
            self.process=subprocess.Popen([sys.executable,'-m','logic_simulator.inference.worker'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,bufsize=0,env=environment)
            self.writer,self.reader=self.process.stdin,self.process.stdout
        else:
            import paramiko
            connection=resolve_environment(target.connection)
            if 'password' in connection:
                raise ValueError('Use password_file or password_prompt; inline passwords are unsupported')
            self.ssh=paramiko.SSHClient()
            self.ssh.load_system_host_keys()
            # Explicitly opt in for session-specific QDC ephemeral host keys.
            if connection.get('accept_new_host_key',False):
                self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            password=None
            if connection.get('password_file'):
                password=Path(connection['password_file']).read_text().rstrip('\r\n')
            elif connection.get('password_prompt'):
                import getpass
                password=getpass.getpass('Board SSH password: ')
            try:
                self.ssh.connect(connection['host'],port=int(connection.get('port',22)),username=connection.get('username','root'),
                    password=password,key_filename=connection.get('key_file'),timeout=15,auth_timeout=15,
                    allow_agent=password is None,look_for_keys=password is None)
            except Exception as error:
                self.ssh.close()
                raise RuntimeError(f'Board SSH connection failed ({type(error).__name__}); check session/route/authentication') from None
            command='docker run --rm -i --network none -v '+shlex.quote(target.model_root+':/models:ro')
            for device in connection.get('devices',[]):
                command+=' --device '+shlex.quote(device)
            if connection.get('gpus'):
                command+=' --gpus '+shlex.quote(str(connection['gpus']))
            command+=' '+shlex.quote(target.worker_image)+' worker'
            self.writer,self.reader,self.stderr=self.ssh.exec_command(command,timeout=timeout)
            self.reader.channel.settimeout(timeout)
        self.backend=None

    def call(self,request,image=None):
        import cv2
        value=dict(request)
        if image is not None:
            success,data=cv2.imencode('.png',image)
            if not success:
                raise ValueError('Cannot encode inference input')
            value['image']=base64.b64encode(data).decode('ascii')
        deadline=time.monotonic()+self.timeout
        send(self.writer,value,timeout=self.timeout)
        result=receive(self.reader,max(0,deadline-time.monotonic()))
        if result.get('request_id') != request['request_id']:
            raise RuntimeError('Worker response request ID mismatch')
        if 'error' in result:
            raise RuntimeError('Worker inference failed: '+result['error'])
        self.backend=result.get('backend')
        return result

    def close(self):
        if self.process:
            self.writer.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.reader.close()
        if self.ssh:
            self.ssh.close()
