# SPDX-License-Identifier: MIT
# Step 3: paste this function into clean_table_scene.py.
# Imports and scene helpers are supplied by the exercise file.

def build_coupled_solver(scene, *, vbd_iterations=20, coupling_iterations=2):
    model = scene.model
    return SolverCoupledProxy(model=model, entries=[
        SolverCoupled.Entry(name='rigid', bodies=scene.rigid_bodies, joints=scene.rigid_joints,
            solver=lambda view: SolverMuJoCo(view, use_mujoco_cpu=not model.device.is_cuda,
                use_mujoco_contacts=True, solver='newton', integrator='implicitfast', cone='elliptic',
                njmax=2048, nconmax=1024, iterations=100, ls_iterations=50, impratio=100)),
        SolverCoupled.Entry(name='deformable', bodies=scene.cable_bodies, joints=scene.cable_joints,
            particles=list(range(model.particle_count)),
            solver=lambda view: ObservedVBD(view, iterations=vbd_iterations, rigid_compliant_alm=True,
                particle_enable_self_contact=True, particle_self_contact_margin=0.002,
                particle_self_contact_gap=0.002, rigid_body_contact_buffer_size=RIGID_CONTACTS_PER_BODY,
                rigid_body_particle_contact_buffer_size=SOFT_CONTACTS_PER_BODY, rigid_contact_history=False)),
    ], coupling=SolverCoupledProxy.Config(iterations=coupling_iterations, proxies=[
        SolverCoupledProxy.Proxy(source='rigid', destination='deformable', bodies=scene.proxy_bodies,
            joints=scene.proxy_joints, mass_scale=1., mode='lagged', proxy_relaxation=0.5,
            collision_pipeline=make_clutter_pipeline, collide_interval=1)
    ]))
