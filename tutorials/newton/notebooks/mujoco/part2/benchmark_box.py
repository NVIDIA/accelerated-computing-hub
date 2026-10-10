# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Measured native MuJoCo/MuJoCo Warp replay of both cubes into a box.

Native worlds run in persistent processes because genuine solved contact
evidence must be collected from live MjData. Both devices record the same
compact observations after a forward pass every 50 Hz control frame.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import warp as wp

from box_task import BoxTask
from robots import get_robot
import pick_place_common as task
from benchmark_box_protocol import (
    FRAMES, SUBSTEPS, DT, OBSERVATION_SIZE, ACCEPTANCE,
    NativeRecorder, CUDARecorder, validate_observations, backend_option_metadata,
    ObservationValidator, HOST_VALIDATION_POLICY,
)

HERE = Path(__file__).resolve().parent
_WORKER = None
_LEGACY = None


def _legacy():
    """Reuse the audited 3.8 capacity guards under an article-specific name."""
    global _LEGACY
    if _LEGACY is None:
        name = "_article2_legacy_benchmark"
        spec = importlib.util.spec_from_file_location(name, HERE / "benchmark_workload.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _LEGACY = module
    return _LEGACY


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _configuration(config):
    values = {"task": "box", "backend": "mujoco", "robot": "so101", "worlds": 1,
              "cpu_threads": 4, "frames": FRAMES, "substeps": SUBSTEPS, "repeats": 5,
              "warmups": 1, "device": "cuda:0", "nconmax": None, "njmax": None,
              "max_trajectory_mib": 8192, **config}
    if values["task"] != "box" or values["backend"] not in ("mujoco", "mjwarp"):
        raise ValueError("Box replay requires task=box and backend=mujoco or mjwarp")
    values.setdefault("validation_workers", values["cpu_threads"])
    get_robot(values["robot"])
    for key in ("worlds", "cpu_threads", "validation_workers", "frames", "substeps", "repeats", "warmups"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["frames"] != FRAMES or values["substeps"] != SUBSTEPS:
        raise ValueError("The box benchmark requires all 2000 frames and 40000 physics steps")
    for key in ("nconmax", "njmax"):
        if values[key] is not None and (type(values[key]) is not int or values[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    limit = float(values["max_trajectory_mib"])
    required = values["worlds"] * FRAMES * OBSERVATION_SIZE * 4
    if not np.isfinite(limit) or limit <= 0 or required > limit * 2**20:
        raise ValueError("Full box observation history exceeds the memory limit; reduce worlds, never task duration")
    if values["backend"] == "mjwarp" and not str(values["device"]).startswith("cuda:"):
        raise ValueError("MJWarp requires an explicit CUDA device; no CPU substitution")
    return values


def _model(config):
    box = BoxTask(get_robot(config["robot"]))
    path = config.get("menagerie_path")
    scene = box.resolve_scene(Path(path) if path else None, explicit=bool(path))
    model = task.load_pick_place_model(scene, box.spec)
    model.opt.timestep = DT
    model.opt.iterations, model.opt.ls_iterations, model.opt.impratio = 100, 50, 100
    model.opt.tolerance = 1e-6  # Match the float32 backend's supported stopping target.
    return model, box, scene


def _prepare(config):
    model, box, scene = _model(config)
    data = mujoco.MjData(model)
    task.apply_arm_ctrl(model, data, box.spec.home_ctrl)
    task.reset_cubes(model, data, box.spec)
    controller = box.make_controller()
    commands, phases = [], []
    for _ in range(FRAMES):
        phases.append(controller.phase_name())
        commands.append(controller.step(model, data, .02).copy())
    tape = np.asarray(commands, dtype=np.float32)
    if not controller.done or not np.isfinite(tape).all():
        raise ValueError("Incomplete or nonfinite box control tape")
    # Seed only once per reset. The original home pose intersects the box/table.
    # Payload coordinates remain at their canonical spawn positions.
    task.apply_arm_ctrl(model, data, tape[0])
    task.reset_cubes(model, data, box.spec)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    names = ("benchmark_box.py", "benchmark_box_protocol.py", "benchmark_workload.py",
             "box_task.py", "pick_place_common.py", "robots.py", "utils.py")
    workload = {
        "protocol_version": 2, "task": "two_cube_pick_place_into_box", "frames": FRAMES, "substeps": SUBSTEPS,
        "physics_steps": FRAMES * SUBSTEPS, "simulated_seconds": 40., "dt": DT,
        "control_dt": .02, "acceptance": ACCEPTANCE,
        "host_validation_policy": {**HOST_VALIDATION_POLICY,
            "requested_workers":config.get("validation_workers",config["cpu_threads"])},
        "controller": "canonical BoxTask precomputed open-loop waypoint/IK tape",
        "control_dtype": "float32 values, cast to float64 by native MuJoCo",
        "initial_robot_ctrl": tape[0].tolist(), "initial_robot_ctrl_sha256": _sha(tape[0].astype('<f4').tobytes()),
        "initialization": "arm at first above-red IK command; canonical cube spawns; reset only",
        "output": "61 float32 fields per world at 50 Hz: actual geometry, corner speeds, opening and solved jaw contacts",
        "observation_policy": "forward after each control frame on the active backend, then measure actual solved contacts",
        "model_dimensions": {key:int(getattr(model,key)) for key in ("nq","nv","nu","na","nbody","ngeom")},
        "model_options": {key:float(getattr(model.opt,key)) for key in
                          ("timestep","iterations","ls_iterations","impratio","tolerance","ls_tolerance")},
        "model_mjb_sha256": _sha(binary.tobytes()), "scene_xml_sha256": _sha(scene.read_bytes()),
        "robot_xml_sha256": _sha((scene.parent / box.spec.robot_xml).read_bytes()),
        "menagerie_ref": box.spec.menagerie_ref,
        "control_tape_sha256": _sha(tape.astype('<f4').tobytes()),
        "initial_state_sha256": _sha(initial.astype('<f8').tobytes()),
        "source_sha256": {name:_sha((HERE/name).read_bytes()) for name in names},
        "simulation_timer": "episode reset, controls, 40000 physics steps, backend forward and real contact/geometry observations; CPU process dispatch/completion or synchronized CUDA",
    }
    workload["comparison_signature"] = _sha(json.dumps(workload,sort_keys=True).encode())
    return model, data, tape, np.asarray(phases), box.spec, workload


class _NativeWorld:
    def __init__(self, config, tape):
        self.model, self.box, _ = _model(config)
        self.data = mujoco.MjData(self.model)
        self.tape = tape
        self.recorder = NativeRecorder(self.model, self.box.spec)

    def episode(self, output):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        task.apply_arm_ctrl(model, data, self.tape[0])
        task.reset_cubes(model, data, self.box.spec)
        integration_count = 0
        for frame in range(FRAMES):
            data.ctrl[:] = self.tape[frame]
            mujoco.mj_step(model, data, nstep=SUBSTEPS)
            integration_count += SUBSTEPS
            mujoco.mj_forward(model, data)
            self.recorder.capture(data, output[frame], frame)
        warnings = {mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all())
        return {"capacity_passed":finite and not warnings, "warnings":warnings,
                "finite_state":finite, "recorded_physics_steps":integration_count,
                "final_time_seconds":float(data.time)}


def _cpu_initialize(config, tape, name, shape, ready, lock):
    global _WORKER
    try:
        with lock:
            world = _NativeWorld(config, tape)
        memory = shared_memory.SharedMemory(name=name)
        output = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
        _WORKER = (world, memory, output)
        ready.put({"passed":True})
    except Exception as exc:
        ready.put({"passed":False,"error":str(exc)})
        raise


def _cpu_episode(world):
    context, _, output = _WORKER
    return {"world":world, **context.episode(output[world])}


class _CPUBatch:
    def __init__(self, config, tape):
        self.worlds = config["worlds"]
        affinity = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
        self.workers = min(self.worlds, config["cpu_threads"], affinity)
        shape = (self.worlds, FRAMES, OBSERVATION_SIZE)
        required = int(np.prod(shape))*4
        if sys.platform.startswith("linux") and Path('/dev/shm').exists():
            stat = os.statvfs('/dev/shm')
            if required > stat.f_bavail*stat.f_frsize:
                raise ValueError("Insufficient shared memory for full box observations")
        self.memory = shared_memory.SharedMemory(create=True,size=required)
        self.output = np.ndarray(shape,dtype=np.float32,buffer=self.memory.buf)
        self.pool = None
        ctx = mp.get_context('spawn')
        self.ready = ctx.Queue()
        overrides = {"CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
                     "MKL_NUM_THREADS":"1", "PXR_WORK_THREAD_LIMIT":"1"}
        prior = {key:os.environ.get(key) for key in overrides}
        # Spawn must resolve this module even when a test imports it from another cwd.
        added = str(HERE) not in sys.path
        if added:
            sys.path.insert(0,str(HERE))
        try:
            os.environ.update(overrides)
            self.pool = ctx.Pool(self.workers,initializer=_cpu_initialize,
                initargs=(config,tape,self.memory.name,shape,self.ready,ctx.Lock()))
            ready = [self.ready.get(timeout=300) for _ in range(self.workers)]
            if not all(row['passed'] for row in ready):
                raise RuntimeError(f"CPU box initialization failed: {ready}")
        except BaseException:
            self.close()
            raise
        finally:
            if added:
                sys.path.remove(str(HERE))
            for key,value in prior.items():
                if value is None:
                    os.environ.pop(key,None)
                else:
                    os.environ[key]=value
        self.info = {"kind":"cpu", "cpu_threads":self.workers, "cpu_workers":self.workers,
                     "cpu_parallelism":"persistent native MuJoCo worker processes",
                     "output_dtype":"float32", "trajectory_bytes":required,
                     "cpu_worker_environment":overrides}

    def episode(self):
        start = time.perf_counter()
        diagnostics = self.pool.map(_cpu_episode,range(self.worlds),chunksize=1)
        elapsed = time.perf_counter()-start
        start = time.perf_counter()
        output = self.output.copy()
        transfer = time.perf_counter()-start
        return output,elapsed,transfer,{"capacity_passed":all(d['capacity_passed'] for d in diagnostics),
                                       "completed_worlds":len(diagnostics),"worlds":diagnostics,
                                       "clock_precision":"float64",
                                       "integration_steps_per_world":[d["recorded_physics_steps"] for d in diagnostics]}

    def close(self):
        if self.pool is not None:
            self.pool.terminate();self.pool.join();self.pool=None
        self.memory.close();self.memory.unlink();self.ready.close()


@wp.kernel(enable_backward=False)
def _load_targets(tape: wp.array2d(dtype=float), cursor: wp.array(dtype=int), ctrl: wp.array2d(dtype=float)):
    world, actuator = wp.tid()
    ctrl[world,actuator] = tape[cursor[0],actuator]


@wp.kernel(enable_backward=False)
def _finite_state(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), bad: wp.array(dtype=int),
                  integration_step: int, step_counts: wp.array(dtype=int)):
    world = wp.tid()
    step_counts[world] += integration_step
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[world,i]):
            bad[world] = 1
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[world,i]):
            bad[world] = 1


class _GPUBatch:
    def __init__(self, config, native, initial, tape, spec):
        import mujoco_warp as mjw
        self.mjw,self.legacy = mjw,_legacy()
        wp.init()
        self.device = wp.get_device(config['device'])
        if not self.device.is_cuda:
            raise ValueError("Requested CUDA device unavailable; no CPU substitution")
        self.worlds = config['worlds']
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(native)
            if hasattr(self.model.opt,'warn_overflow'):
                self.model.opt.warn_overflow=False
            self.data = mjw.make_data(native,nworld=self.worlds,nconmax=config['nconmax'] or spec.nconmax,
                njmax=config['njmax'] or spec.njmax,njmax_nnz=(config['njmax'] or spec.njmax)*native.nv)
            self.capacity_mode = self.legacy._capacity_mode(mjw,native,self.data)
            if self.data.time.dtype != wp.float32:
                raise ValueError("Clock contract requires actual float32 MuJoCo Warp time")
            self.initial_qpos = wp.array(np.tile(initial.qpos,(self.worlds,1)),dtype=float)
            self.initial_qvel = wp.array(np.tile(initial.qvel,(self.worlds,1)),dtype=float)
            self.initial_ctrl = wp.array(np.tile(initial.ctrl,(self.worlds,1)),dtype=float)
            self.tape = wp.array(tape,dtype=float)
            self.cursor = wp.zeros(1,dtype=int)
            self.bad = wp.zeros(self.worlds,dtype=int)
            self.step_counts = wp.zeros(self.worlds,dtype=int)
            self.contact_peak = wp.zeros(1,dtype=int)
            self.collision_peak = wp.zeros(1,dtype=int)
            self.constraint_peak = wp.zeros(self.worlds,dtype=int)
            self.recorder = CUDARecorder(native,self.model,self.data,spec)
            self.reset();self.frame();self.reset()
            with wp.ScopedCapture(device=self.device) as capture:
                self.frame()
            self.graph=capture.graph
            wp.synchronize_device(self.device)
            self.info = {"kind":"cuda","name":self.device.name,"alias":self.device.alias,
                "architecture":int(self.device.arch),"output_dtype":"float32",
                "trajectory_bytes":self.recorder.output.size*4,"naconmax":int(self.data.naconmax),
                "njmax":int(self.data.njmax),"naccdmax":int(self.data.naccdmax),
                "capacity_monitor":self.capacity_mode,
                "backend_options":backend_option_metadata(native,self.model)}

    def capacity(self, integration_step=0):
        d=self.data
        wp.launch(self.legacy._record_capacity,self.worlds,inputs=[d.nacon,d.nefc,d.ncollision,
            self.contact_peak,self.constraint_peak,self.collision_peak])
        wp.launch(_finite_state,self.worlds,inputs=[d.qpos,d.qvel,self.bad,integration_step,self.step_counts])

    def reset(self):
        d=self.data
        self.mjw.reset_data(self.model,d)
        wp.copy(d.qpos,self.initial_qpos);wp.copy(d.qvel,self.initial_qvel);wp.copy(d.ctrl,self.initial_ctrl)
        self.cursor.zero_();self.bad.zero_();self.step_counts.zero_();self.contact_peak.zero_();self.constraint_peak.zero_();self.collision_peak.zero_()
        self.mjw.forward(self.model,d);self.capacity();d.qacc_warmstart.zero_()

    def frame(self):
        wp.launch(_load_targets,(self.worlds,self.tape.shape[1]),inputs=[self.tape,self.cursor,self.data.ctrl])
        for _ in range(SUBSTEPS):
            self.mjw.step(self.model,self.data);self.capacity(integration_step=1)
        self.mjw.forward(self.model,self.data);self.capacity()
        self.recorder.record(self.cursor)
        wp.launch(self.legacy._advance_cursor,1,inputs=[self.cursor])

    def episode(self):
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            start=time.perf_counter();self.reset()
            for _ in range(FRAMES):
                wp.capture_launch(self.graph)
            wp.synchronize_device(self.device);elapsed=time.perf_counter()-start
            start=time.perf_counter()
            output=self.recorder.output.numpy()
            flags=self.data.overflow.numpy() if hasattr(self.data,'overflow') else None
            contacts=int(self.contact_peak.numpy()[0]);constraints=self.constraint_peak.numpy()
            collisions=int(self.collision_peak.numpy()[0]);cursor=int(self.cursor.numpy()[0]);bad=self.bad.numpy()
            step_counts=self.step_counts.numpy()
            transfer=time.perf_counter()-start
        iterations=int(self.mjw.OverflowType.ITERATIONS|self.mjw.OverflowType.LS_ITERATIONS) if flags is not None else 0
        diagnostic=self.legacy._capacity_diagnostics(flags,contacts,constraints,iteration_mask=iterations,
            naconmax=self.data.naconmax,njmax=self.data.njmax,cursor=cursor*SUBSTEPS,steps=FRAMES*SUBSTEPS,
            collisions=collisions,mode=self.capacity_mode)
        diagnostic['nonfinite_state_per_world']=bad.tolist()
        diagnostic['integration_steps_per_world']=step_counts.tolist()
        diagnostic['clock_precision']='float32'
        diagnostic['capacity_passed']=bool(diagnostic['capacity_passed'] and not np.any(bad))
        return output,elapsed,transfer,diagnostic

    def close(self):
        pass


def run_case(config):
    started=time.perf_counter()
    row={"status":"failed","config":dict(config),"samples":[],"warmup_samples":[],"timings":{},"workload":{},"diagnostics":{}}
    engine=validator=None
    try:
        config=_configuration(config)
        row.update(config=config,backend=config['backend'],robot=config['robot'],worlds=config['worlds'])
        model,initial,tape,phases,spec,row['workload']=_prepare(config)
        row['timings']['preparation_seconds']=time.perf_counter()-started
        start=time.perf_counter()
        engine=_CPUBatch(config,tape) if config['backend']=='mujoco' else _GPUBatch(config,model,initial,tape,spec)
        row['timings']['backend_setup_seconds']=time.perf_counter()-start
        row['device_info']=engine.info
        if config['backend']=='mujoco':
            row['device_info']['backend_options']=backend_option_metadata(model)
        row['backend_settings']=row['device_info']['backend_options']
        start=time.perf_counter()
        validator=ObservationValidator(spec,phases,config['worlds'],config['validation_workers'])
        row['timings']['validation_setup_seconds']=time.perf_counter()-start
        row['host_validation']=validator.info
        for warmup in (True,False):
            group=time.perf_counter()
            for repeat in range(config['warmups'] if warmup else config['repeats']):
                history,elapsed,transfer,diagnostics=engine.episode()
                start=time.perf_counter();validation=validator.validate(history,
                    clock_precision=diagnostics["clock_precision"],
                    step_counts=diagnostics["integration_steps_per_world"])
                checked=time.perf_counter()-start
                passed=bool(validation['passed'] and diagnostics['capacity_passed'])
                sample={"repeat":repeat,"simulation_seconds":elapsed,"output_transfer_seconds":transfer,
                    "validation_seconds":checked,"passed":passed,"task_success_count":validation['task_success_count'],
                    "task_total_count":config['worlds'],"validation":validation,"diagnostics":diagnostics}
                row['warmup_samples' if warmup else 'samples'].append(sample)
                if not passed:
                    raise ValueError("Full two-cube box episode failed physical evidence or capacity checks")
            if warmup:
                row['timings']['warmup_seconds']=time.perf_counter()-group
        row['status']='passed'
    except Exception as exc:
        row['diagnostics'].update(error_type=type(exc).__name__,error=str(exc))
    finally:
        start=time.perf_counter()
        try:
            if validator is not None:
                validator.close()
        finally:
            row['timings']['validation_teardown_seconds']=time.perf_counter()-start
            try:
                if engine is not None:
                    engine.close()
            finally:
                row['timings']['case_seconds']=time.perf_counter()-started
    return row
