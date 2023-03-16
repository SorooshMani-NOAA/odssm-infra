# Run from prefect directory (after terraform vars gen) using
# prefect run --name sim-prep --param name=florance --param year=2018


# For logging, use `logger = prefect.context.get("logger")` within tasks
import argparse
import warnings
import pathlib

from prefect import case
from prefect.utilities import graphql
from prefect.client import Client
from prefect.tasks.control_flow import merge

from conf import PREFECT_PROJECT_NAME, INIT_FINI_LOCK
from tasks.params import (
        param_storm_name, param_storm_year,
        param_use_rdhpcs, param_use_rdhpcs_post, param_run_id,
        param_use_parametric_wind, param_subset_mesh, param_ensemble,
        param_mesh_hmax,
        param_mesh_hmin_low, param_mesh_rate_low,
        param_mesh_trans_elev,
        param_mesh_hmin_high, param_mesh_rate_high,
        param_ensemble_n_perturb, param_hr_prelandfall,
        param_ensemble_sample_rule,
        param_past_forecast,
        )
from tasks.data import (
        task_copy_s3_data, 
        task_init_run,
        task_final_results_to_s3,
        task_cleanup_run,
        task_cache_to_s3,
        task_cleanup_efs)
from tasks.utils import (
        task_check_param_true,
        task_bundle_params,
        task_get_flow_run_id,
        task_get_run_tag,
        FLock)
from flows.jobs.ecs import (
        make_flow_generic_ecs_task,
        make_flow_solve_ecs_task
        )
from flows.jobs.pw import(
        make_flow_mesh_rdhpcs_pw_task,
        make_flow_mesh_rdhpcs,
        make_flow_solve_rdhpcs_pw_task,
        make_flow_solve_rdhpcs)
from flows.utils import LocalAWSFlow, flow_dependency


# TODO: Later add build image and push to ECS logic into Prefect workflow

# TODO: Use subprocess.run to switch backend here
# TODO: Create user config file to be session based? https://docs-v1.prefect.io/core/concepts/configuration.html#environment-variables

def _check_project():
    client = Client()
    print(f"Connecting to {client.api_server}...")

    qry = graphql.parse_graphql({'query': {'project': ['name']}})
    rsp = client.graphql(qry)

    prj_names = [i['name'] for i in rsp['data']['project']]
    if PREFECT_PROJECT_NAME in prj_names:
        print(f"Project {PREFECT_PROJECT_NAME} found on {client.api_server}!")
        return

    print(f"Creating project {PREFECT_PROJECT_NAME} on {client.api_server}...")
    client.create_project(project_name=PREFECT_PROJECT_NAME)
    print("Done!")


def _make_workflow():
    # Create flow objects
    flow_sim_prep_info_aws = make_flow_generic_ecs_task("sim-prep-info-aws")
    flow_sim_prep_mesh_aws = make_flow_generic_ecs_task("sim-prep-mesh-aws")
    flow_sim_prep_setup_aws = make_flow_generic_ecs_task("sim-prep-setup-aws")
    flow_mesh_rdhpcs_pw_task = make_flow_mesh_rdhpcs_pw_task()
    flow_mesh_rdhpcs = make_flow_mesh_rdhpcs(flow_mesh_rdhpcs_pw_task)
    flow_schism_single_run_aws = make_flow_generic_ecs_task("schism-run-aws-single")
    flow_schism_ensemble_run_aws = make_flow_solve_ecs_task(flow_schism_single_run_aws)
    flow_solve_rdhpcs_pw_task = make_flow_solve_rdhpcs_pw_task()
    flow_solve_rdhpcs = make_flow_solve_rdhpcs(flow_solve_rdhpcs_pw_task)
    flow_sta_html_aws = make_flow_generic_ecs_task("viz-sta-html-aws")
    flow_cmb_ensemble_aws = make_flow_generic_ecs_task(
        "viz-cmb-ensemble-aws"
    )
    flow_ana_ensemble_aws = make_flow_generic_ecs_task(
        "viz-ana-ensemble-aws"
    )


    with LocalAWSFlow("end-to-end") as flow_main:

        result_flow_run_id = task_get_flow_run_id()

        result_run_tag = task_get_run_tag(
            param_storm_name, param_storm_year, result_flow_run_id)

        result_is_rdhpcs_on = task_check_param_true(param_use_rdhpcs)
        result_is_ensemble_on = task_check_param_true(param_ensemble)
        result_is_rdhpcspost_on = task_check_param_true(param_use_rdhpcs_post)

        with FLock(INIT_FINI_LOCK, task_args={'name': 'Sync init'}):
            result_copy_task = task_copy_s3_data()
            result_init_run = task_init_run(
                    result_run_tag, upstream_tasks=[result_copy_task])

        result_bundle_params_1 = task_bundle_params(
                name=param_storm_name,
                year=param_storm_year,
                rdhpcs=param_use_rdhpcs,
                rdhpcs_post=param_use_rdhpcs_post,
                run_id=result_flow_run_id,
                parametric_wind=param_use_parametric_wind,
                ensemble=param_ensemble,
                hr_before_landfall=param_hr_prelandfall,
                past_forecast=param_past_forecast,
                )

        result_bundle_params_2 = task_bundle_params(
                name=param_storm_name,
                year=param_storm_year,
                rdhpcs=param_use_rdhpcs,
                run_id=result_flow_run_id,
                subset_mesh=param_subset_mesh,
                mesh_hmax=param_mesh_hmax,
                mesh_hmin_low=param_mesh_hmin_low,
                mesh_rate_low=param_mesh_rate_low,
                mesh_cutoff=param_mesh_trans_elev,
                mesh_hmin_high=param_mesh_hmin_high,
                mesh_rate_high=param_mesh_rate_high
                )

        result_bundle_params_3 = task_bundle_params(
                name=param_storm_name,
                year=param_storm_year,
                run_id=result_flow_run_id,
                parametric_wind=param_use_parametric_wind,
                ensemble=param_ensemble,
                ensemble_num_perturbations=param_ensemble_n_perturb,
                hr_before_landfall=param_hr_prelandfall,
                ensemble_sample_rule=param_ensemble_sample_rule,
        )

        after_sim_prep_info = flow_dependency(
                flow_name=flow_sim_prep_info_aws.name,
                upstream_tasks=[result_init_run],
                parameters=result_bundle_params_1)

        # TODO: Meshing based-on original track for now
        # TODO: If mesh each track: diff mesh


        with case(result_is_rdhpcs_on, True):
            after_sim_prep_mesh_b1 = flow_dependency(
                    flow_name=flow_mesh_rdhpcs.name,
                    upstream_tasks=[after_sim_prep_info],
                    parameters=result_bundle_params_2)
        with case(result_is_rdhpcs_on, False):
            after_sim_prep_mesh_b2 = flow_dependency(
                    flow_name=flow_sim_prep_mesh_aws.name,
                    upstream_tasks=[after_sim_prep_info],
                    parameters=result_bundle_params_2)
        after_sim_prep_mesh = merge(after_sim_prep_mesh_b1, after_sim_prep_mesh_b2)

        after_sim_prep_setup = flow_dependency(
                flow_name=flow_sim_prep_setup_aws.name,
                upstream_tasks=[after_sim_prep_mesh],
                parameters=result_bundle_params_3)

        with case(result_is_rdhpcs_on, True):
            after_run_schism_b1 = flow_dependency(
                    flow_name=flow_solve_rdhpcs.name,
                    upstream_tasks=[after_sim_prep_setup],
                    parameters=result_bundle_params_1)
        with case(result_is_rdhpcs_on, False):
            after_run_schism_b2 = flow_dependency(
                flow_name=flow_schism_ensemble_run_aws.name,
                upstream_tasks=[after_sim_prep_setup],
                parameters=result_bundle_params_1)
        after_run_schism = merge(after_run_schism_b1, after_run_schism_b2)


        with case(result_is_ensemble_on, True):
            with case(result_is_rdhpcspost_on, False):
                after_cmb_ensemble = flow_dependency(
                    flow_name=flow_cmb_ensemble_aws.name,
                    upstream_tasks=[after_run_schism],
                    parameters=result_bundle_params_1)
                after_ana_ensemble = flow_dependency(
                    flow_name=flow_ana_ensemble_aws.name,
                    upstream_tasks=[after_cmb_ensemble],
                    parameters=result_bundle_params_1)

            with case(result_is_rdhpcspost_on, True):
                # TODO:
                pass

        with case(result_is_ensemble_on, False):
            after_sta_html = flow_dependency(
                flow_name=flow_sta_html_aws.name,
                upstream_tasks=[after_run_schism],
                parameters=result_bundle_params_1)
        after_gen_viz = merge(after_ana_ensemble, after_sta_html)
        
        # TODO: Make this a separate flow?
        after_results_to_s3 = task_final_results_to_s3(
                param_storm_name, param_storm_year, result_run_tag,
                upstream_tasks=[after_gen_viz])

        after_cleanup_run = task_cleanup_run(
                result_run_tag, upstream_tasks=[after_results_to_s3])

        with FLock(INIT_FINI_LOCK, upstream_tasks=[after_cleanup_run], task_args={'name': 'Sync cleanup'}):
            after_cache_storage = task_cache_to_s3(
                    upstream_tasks=[after_cleanup_run])
            task_cleanup_efs(
                    result_run_tag,
                    upstream_tasks=[after_cache_storage])

    flow_main.set_reference_tasks([after_cleanup_run])

    all_flows = [
        flow_sim_prep_info_aws,
        flow_sim_prep_mesh_aws,
        flow_sim_prep_setup_aws,
        flow_mesh_rdhpcs_pw_task,
        flow_mesh_rdhpcs,
        flow_schism_single_run_aws,
        flow_schism_ensemble_run_aws,
        flow_solve_rdhpcs_pw_task,
        flow_solve_rdhpcs,
        flow_sta_html_aws,
        flow_cmb_ensemble_aws,
        flow_ana_ensemble_aws,
        flow_main
    ]

    return all_flows

def _regiser(flows):
    # Register unregistered flows
    for flow in flows:
        flow.register(project_name=PREFECT_PROJECT_NAME)

def _viz(flows, out_dir, flow_names):
    flow_dict = {f.name: f for f in flows}
    for flow_nm in flow_names:
        flow = flow_dict.get(flow_nm)
        if flow is None:
            warnings.warn(f'Flow with the name {flow_nm} NOT found!')
        flow.visualize(filename=out_dir/flow.name, format='dot')

def _list(flows):
    flow_names = [f.name for f in flows]
    print("\n".join(flow_names))


def _main(args):

    _check_project()
    all_flows = _make_workflow()
    if args.command in ["register", None]:
        _regiser(all_flows)

    elif args.command == "visualize":
        _viz(all_flows, args.out_dir, args.flowname)

    elif args.command == "list":
        _list(all_flows)

    else:
        raise ValueError("Invalid command!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    reg_parser = subparsers.add_parser('register')
    viz_parser = subparsers.add_parser('visualize')
    list_parser = subparsers.add_parser('list')

    viz_parser.add_argument('flowname', nargs='+')
    viz_parser.add_argument(
        '--out-dir', '-d', type=pathlib.Path, default='.')
    
    _main(parser.parse_args())
