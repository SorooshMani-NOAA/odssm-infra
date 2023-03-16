from functools import partial

from prefect import apply_map, unmapped, case, task
from prefect.utilities.edges import unmapped
from prefect.tasks.secrets import EnvVarSecret
from prefect.tasks.files.operations import Glob
from prefect.tasks.prefect.flow_run import create_flow_run, wait_for_flow_run

from conf import (
    OCSMESH_CLUSTER, OCSMESH_TEMPLATE_1_ID, OCSMESH_TEMPLATE_2_ID,
    SCHISM_CLUSTER, SCHISM_TEMPLATE_ID,
    VIZ_CLUSTER, VIZ_TEMPLATE_ID,
    WF_CLUSTER, WF_TEMPLATE_ID, WF_IMG,
    ECS_TASK_ROLE, ECS_EXEC_ROLE,
    PREFECT_PROJECT_NAME,
)
from tasks.params import (
    param_storm_name, param_storm_year, param_run_id,
    param_use_parametric_wind, param_schism_dir,
    param_subset_mesh, param_ensemble,
    param_mesh_hmax,
    param_mesh_hmin_low, param_mesh_rate_low,
    param_mesh_trans_elev,
    param_mesh_hmin_high, param_mesh_rate_high,
    param_ensemble_n_perturb, param_hr_prelandfall,
    param_ensemble_sample_rule,
    param_past_forecast,
)
from tasks.infra import ContainerInstance, task_add_ecs_attribute_for_ec2
from tasks.jobs import (
    task_start_ecs_task,
    task_format_start_task,
    shell_run_task,
    task_client_wait_for_ecs,
    task_retrieve_task_docker_logs,
    task_kill_task_if_wait_fails,
    task_format_kill_timedout,
    task_check_docker_success)
from tasks.utils import (
        ECSTaskDetail,
        task_check_param_true,
        task_pylist_from_jsonlist,
        task_get_run_tag,
        task_get_flow_run_id,
        task_bundle_params,
        task_replace_tag_in_template,
        task_convert_str_to_path,
        task_return_value_if_param_true,
        task_return_value_if_param_false)
from flows.utils import LocalAWSFlow, flow_dependency, task_create_ecsrun_config



def _use_if(param, is_true, value):
    if is_true:
        task = task_return_value_if_param_true
    else:
        task = task_return_value_if_param_false

    return lambda: task(param=param, value=value)


def _tag(template):
    return lambda: task_replace_tag_in_template(
            storm_name=param_storm_name,
            storm_year=param_storm_year,
            run_id=param_run_id,
            template_str=str(template))


def _tag_n_use_if(param, is_true, template):
    if is_true:
        task = task_return_value_if_param_true
    else:
        task = task_return_value_if_param_false

    return lambda: task(
                param=param,
                value=task_replace_tag_in_template(
                    storm_name=param_storm_name,
                    storm_year=param_storm_year,
                    run_id=param_run_id,
                    template_str=str(template)
                )
            )


def _use_if_and(*and_conds, value=None):
    assert value is not None
    assert len(and_conds) % 2 == 0
    tasks_args = []
    conds_iter = iter(and_conds)
    for par, is_true in zip(conds_iter, conds_iter):
        if is_true:
            tasks_args.append((task_return_value_if_param_true, par))
        else:
            tasks_args.append((task_return_value_if_param_false, par))

    def _call_task(remains):
        assert len(remains) > 0
        task_arg = remains[0]
        if len(remains) == 1:
            return task_arg[0](param=task_arg[1], value=value)
        return task_arg[0](param=task_arg[1], value=_call_task(remains[1:]))

    def _task_recurse():
        return _call_task(tasks_args)

    return _task_recurse


def _tag_n_use_if_and(*and_conds, template=None):
    assert template is not None
    assert len(and_conds) % 2 == 0
    tasks_args = []
    conds_iter = iter(and_conds)
    for par, is_true in zip(conds_iter, conds_iter):
        if is_true:
            tasks_args.append((task_return_value_if_param_true, par))
        else:
            tasks_args.append((task_return_value_if_param_false, par))

    def _call_task(remains):
        assert len(remains) > 0
        task_arg = remains[0]
        if len(remains) == 1:
            return task_arg[0](
                    param=task_arg[1],
                    value=task_replace_tag_in_template(
                        storm_name=param_storm_name,
                        storm_year=param_storm_year,
                        run_id=param_run_id,
                        template_str=str(template)
                    )
                )
        return task_arg[0](param=task_arg[1], value=_call_task(remains[1:]))

    def _task_recurse():
        return _call_task(tasks_args)

    return _task_recurse


info_flow_ecs_task_details = {
    "sim-prep-info-aws": ECSTaskDetail(
        OCSMESH_CLUSTER, OCSMESH_TEMPLATE_2_ID, "odssm-info", "info",
        [
            '--date-range-outpath',
            _tag('{tag}/setup/dates.csv'),
            '--track-outpath',
            _tag('{tag}/nhc_track/hurricane-track.dat'),
            '--swath-outpath',
            _tag('{tag}/windswath'),
            '--station-data-outpath',
            _tag('{tag}/coops_ssh/stations.nc'),
            '--station-location-outpath',
            _tag('{tag}/setup/stations.csv'),
            _use_if(param_past_forecast, True, '--past-forecast'),
            _use_if(param_past_forecast, True, "--hours-before-landfall"),
            _use_if(param_past_forecast, True, param_hr_prelandfall),
            param_storm_name, param_storm_year,
        ],
        "hurricane info",
        60, 20, []),
    "sim-prep-mesh-aws": ECSTaskDetail(
        OCSMESH_CLUSTER, OCSMESH_TEMPLATE_1_ID, "odssm-mesh", "mesh", [
            param_storm_name, param_storm_year,
            "--rasters-dir", 'dem',
            # If subsetting flag is False
            _use_if(param_subset_mesh, False, "hurricane_mesh"),
            _use_if(param_subset_mesh, False, "--hmax"),
            _use_if(param_subset_mesh, False, param_mesh_hmax),
            _use_if(param_subset_mesh, False, "--hmin-low"),
            _use_if(param_subset_mesh, False, param_mesh_hmin_low),
            _use_if(param_subset_mesh, False, "--rate-low"),
            _use_if(param_subset_mesh, False, param_mesh_rate_low),
            _use_if(param_subset_mesh, False, "--transition-elev"),
            _use_if(param_subset_mesh, False, param_mesh_trans_elev),
            _use_if(param_subset_mesh, False, "--hmin-high"),
            _use_if(param_subset_mesh, False, param_mesh_hmin_high),
            _use_if(param_subset_mesh, False, "--rate-high"),
            _use_if(param_subset_mesh, False, param_mesh_rate_high),
            _use_if(param_subset_mesh, False, "--shapes-dir"),
            _use_if(param_subset_mesh, False, 'shape'),
            _use_if(param_subset_mesh, False, "--windswath"),
            _tag_n_use_if(
                param_subset_mesh, False, 'hurricanes/{tag}/windswath'
            ),
            # If subsetting flag is True
            _use_if(param_subset_mesh, True, "subset_n_combine"),
            _use_if(param_subset_mesh, True, 'grid/HSOFS_250m_v1.0_fixed.14'),
            _use_if(param_subset_mesh, True, 'grid/WNAT_1km.14'),
            _tag_n_use_if(
                param_subset_mesh, True, 'hurricanes/{tag}/windswath'
            ),
            # Other shared options
            "--out", _tag('hurricanes/{tag}/mesh'),
        ],
        "meshing",
        60, 180, []),
    "sim-prep-setup-aws": ECSTaskDetail(
        OCSMESH_CLUSTER, OCSMESH_TEMPLATE_2_ID, "odssm-prep", "prep", [
            # Command and arguments for deterministic run
            _use_if(param_ensemble, False, "setup_model"),
            _use_if_and(
                param_use_parametric_wind, True, param_ensemble, False,
                value="--parametric-wind"
            ),
            _use_if(param_ensemble, False, "--mesh-file"),
            _tag_n_use_if(
                param_ensemble, False, 'hurricanes/{tag}/mesh/mesh_w_bdry.grd'
            ),
            _use_if(param_ensemble, False, "--domain-bbox-file"),
            _tag_n_use_if(
                param_ensemble, False, 'hurricanes/{tag}/mesh/domain_box/'
            ),
            _use_if(param_ensemble, False, "--station-location-file"),
            _tag_n_use_if(
                param_ensemble, False, 'hurricanes/{tag}/setup/stations.csv'
            ),
            _use_if(param_ensemble, False, "--out"),
            _tag_n_use_if(
                param_ensemble, False, 'hurricanes/{tag}/setup/schism.dir/'
            ),
            _use_if_and(
                param_use_parametric_wind, True, param_ensemble, False,
                value="--track-file"
            ),
            _tag_n_use_if_and(
                param_use_parametric_wind, True, param_ensemble, False,
                template='hurricanes/{tag}/nhc_track/hurricane-track.dat',
            ),
            _use_if(param_ensemble, False, "--cache-dir"),
            _use_if(param_ensemble, False, 'cache'),
            _use_if(param_ensemble, False, "--nwm-dir"),
            _use_if(param_ensemble, False, 'nwm'),
            # Command and arguments for ensemble run
            _use_if(param_ensemble, True, "setup_ensemble"),
            _use_if(param_ensemble, True, "--track-file"),
            _tag_n_use_if(
                param_ensemble, True, 'hurricanes/{tag}/nhc_track/hurricane-track.dat',
            ),
            _use_if(param_ensemble, True, "--output-directory"),
            _tag_n_use_if(
                param_ensemble, True, 'hurricanes/{tag}/setup/ensemble.dir/'
            ),
            _use_if(param_ensemble, True, "--num-perturbations"),
            _use_if(param_ensemble, True, param_ensemble_n_perturb),
            _use_if(param_ensemble, True, '--mesh-directory'),
            _tag_n_use_if(
                param_ensemble, True, 'hurricanes/{tag}/mesh/'
            ),
            _use_if(param_ensemble, True, "--sample-from-distribution"),
#           _use_if(param_ensemble, True, "--quadrature"),
            _use_if(param_ensemble, True, "--sample-rule"),
            _use_if(param_ensemble, True, param_ensemble_sample_rule),
            _use_if(param_ensemble, True, "--hours-before-landfall"),
            _use_if(param_ensemble, True, param_hr_prelandfall),
            _use_if(param_ensemble, True, "--nwm-file"),
            _use_if(param_ensemble, 
                True,
                "nwm/NWM_v2.0_channel_hydrofabric/nwm_v2_0_hydrofabric.gdb"
            ),
            # Common arguments
            "--date-range-file",
            _tag('hurricanes/{tag}/setup/dates.csv'),
            "--tpxo-dir", 'tpxo',
            param_storm_name, param_storm_year],
        "setup",
        60, 180, ["CDSAPI_URL", "CDSAPI_KEY"]),
    "schism-run-aws-single": ECSTaskDetail(
        SCHISM_CLUSTER, SCHISM_TEMPLATE_ID, "odssm-solve", "solve", [
            param_schism_dir
        ],
        "SCHISM",
        60, 240, []),
    "viz-sta-html-aws": ECSTaskDetail(
        VIZ_CLUSTER, VIZ_TEMPLATE_ID, "odssm-post", "post", [
            param_storm_name, param_storm_year,
            _tag('hurricanes/{tag}/setup/schism.dir/'),
            ],
        "visualization",
        20, 45, []),
    "viz-cmb-ensemble-aws": ECSTaskDetail(
        SCHISM_CLUSTER, SCHISM_TEMPLATE_ID, "odssm-prep", "prep", [
            'combine_ensemble',
            '--ensemble-dir', 
            _tag('hurricanes/{tag}/setup/ensemble.dir/'),
            '--tracks-dir', 
            _tag('hurricanes/{tag}/setup/ensemble.dir/track_files'),
            ],
        "Combine ensemble output files",
        60, 90, []),
    "viz-ana-ensemble-aws": ECSTaskDetail(
        SCHISM_CLUSTER, SCHISM_TEMPLATE_ID, "odssm-prep", "prep", [
            'analyze_ensemble',
            '--ensemble-dir', 
            _tag('hurricanes/{tag}/setup/ensemble.dir/'),
            '--tracks-dir', 
            _tag('hurricanes/{tag}/setup/ensemble.dir/track_files'),
            ],
        "Analyze combined ensemble output",
        60, 90, []),
}

def helper_call_prefect_task_for_ecs_job(
        cluster_name,
        ec2_template,
        description,
        name_ecs_task,
        name_docker,
        command,
        wait_delay=60,
        wait_attempt=150,
        environment=None):

    additional_kwds = {}
    if environment is not None:
        env = additional_kwds.setdefault('env', [])
        for item in environment:
            env.append({
                "name": item,
                "value": EnvVarSecret(item, raise_if_missing=True)}
            )


    # Using container instance per ecs flow, NOT main flow
    thisflow_run_id = task_get_flow_run_id()
    with ContainerInstance(thisflow_run_id, ec2_template):

        result_ecs_task = task_start_ecs_task(
            task_args=dict(
                name=f'Start {description}',
                ),
            command=task_format_start_task(
                template=shell_run_task,
                cluster=cluster_name,
                docker_cmd=command,
                name_ecs_task=name_ecs_task,
                name_docker=name_docker,
                run_tag=thisflow_run_id,
                **additional_kwds)
        )

        result_wait_ecs = task_client_wait_for_ecs(
            waiter_kwargs=dict(
                cluster=cluster_name,
                tasks=task_pylist_from_jsonlist(result_ecs_task),
                WaiterConfig=dict(Delay=wait_delay, MaxAttempts=wait_attempt)
            )
        )

        task_retrieve_task_docker_logs(
                tasks=task_pylist_from_jsonlist(result_ecs_task),
                log_prefix=name_ecs_task,
                container_name=name_docker,
                upstream_tasks=[result_wait_ecs])

        # Timeout based on Prefect wait
        task_kill_task_if_wait_fails.map(
            upstream_tasks=[unmapped(result_wait_ecs)],
            command=task_format_kill_timedout.map(
                    cluster=unmapped(cluster_name),
                    task=task_pylist_from_jsonlist(result_ecs_task)
            )
        )

        result_docker_success = task_check_docker_success(
                upstream_tasks=[result_wait_ecs],
                cluster_name=cluster_name,
                tasks=task_pylist_from_jsonlist(result_ecs_task))

    return result_docker_success



@task
def _task_pathlist_to_strlist(path_list, rel_to=None):
    '''PosixPath objects are not picklable and need to be converted to string'''
    return [str(p) if rel_to is None else str(p.relative_to(rel_to)) for p in path_list]



def make_flow_generic_ecs_task(flow_name):

    task_detail = info_flow_ecs_task_details[flow_name]

    with LocalAWSFlow(flow_name) as flow:
        ref_task = helper_call_prefect_task_for_ecs_job(
            cluster_name=task_detail.name_ecs_cluster,
            ec2_template=task_detail.id_ec2_template,
            description=task_detail.description,
            name_ecs_task=task_detail.name_ecs_task,
            name_docker=task_detail.name_docker,
            wait_delay=task_detail.wait_delay,
            wait_attempt=task_detail.wait_max_attempt,
            environment=task_detail.env_secrets,
            command=[
                i() if callable(i) else i
                for i in task_detail.docker_args])

    flow.set_reference_tasks([ref_task])
    return flow


def make_flow_solve_ecs_task(child_flow):


    ref_tasks = []
    with LocalAWSFlow("schism-run-aws-ensemble") as flow:

        result_is_ensemble_on = task_check_param_true(param_ensemble)
        with case(result_is_ensemble_on, False):
            rundir = task_replace_tag_in_template(
                storm_name=param_storm_name,
                storm_year=param_storm_year,
                run_id=param_run_id,
                template_str='hurricanes/{tag}/setup/schism.dir/'
            )

            ref_tasks.append(
                flow_dependency(
                    flow_name=child_flow.name,
                    upstream_tasks=None,
                    parameters=task_bundle_params(
                        name=param_storm_name,
                        year=param_storm_year,
                        run_id=param_run_id,
                        schism_dir=rundir
                    )
                )
            )

        with case(result_is_ensemble_on, True):
            result_ensemble_dir = task_replace_tag_in_template(
                storm_name=param_storm_name,
                storm_year=param_storm_year,
                run_id=param_run_id,
                template_str='hurricanes/{tag}/setup/ensemble.dir/'
            )

            run_tag = task_get_run_tag(
                    storm_name=param_storm_name,
                    storm_year=param_storm_year,
                    run_id=param_run_id)

            # Start an EC2 to manage ensemble flow runs
            with ContainerInstance(run_tag, WF_TEMPLATE_ID) as ec2_ids:

                task_add_ecs_attribute_for_ec2(ec2_ids, WF_CLUSTER, run_tag)
                ecs_config = task_create_ecsrun_config(run_tag)
                coldstart_task = flow_dependency(
                    flow_name=child_flow.name,
                    upstream_tasks=None,
                    parameters=task_bundle_params(
                        name=param_storm_name,
                        year=param_storm_year,
                        run_id=param_run_id,
                        schism_dir=result_ensemble_dir + '/spinup'
                    ),
                    run_config=ecs_config,
                )
                
                hotstart_dirs = Glob(pattern='runs/*')(
                    path=task_convert_str_to_path('/efs/' + result_ensemble_dir)
                )

                flow_run_uuid = create_flow_run.map(
                    flow_name=unmapped(child_flow.name),
                    project_name=unmapped(PREFECT_PROJECT_NAME),
                    parameters=task_bundle_params.map(
                        name=unmapped(param_storm_name),
                        year=unmapped(param_storm_year),
                        run_id=unmapped(param_run_id),
                        schism_dir=_task_pathlist_to_strlist(
                            hotstart_dirs, rel_to='/efs'
                        )
                    ),
                    upstream_tasks=[unmapped(coldstart_task)],
                    run_config=unmapped(ecs_config)
                )

                hotstart_task = wait_for_flow_run.map(
                    flow_run_uuid, raise_final_state=unmapped(True))


            ref_tasks.append(coldstart_task)
            ref_tasks.append(hotstart_task)

    flow.set_reference_tasks(ref_tasks)
    return flow
