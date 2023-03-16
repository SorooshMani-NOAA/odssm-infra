terraform {
  backend "s3" {
    bucket = "tacc-nos-icogs-backend"
    key    = "terraform/state"
    region = "us-east-1"
  }
}

locals {
  common_tags = {
    Name = "On-Demand Storm Surge Modeling"
    Phase = "Development"
    POCName = "saeed.moghimi@noaa.gov"
    Project = "NOAA ICOGS-C"
    LineOffice = "NOS"
    DivisionBranch = "CSDL-CMMB"
  }
  docker_user = "ondemand-user"
  task_role_arn = "arn:aws:iam::${var.account_id}:role/${var.role_prefix}_ECS_Role"
  execution_role_arn = "arn:aws:iam::${var.account_id}:role/${var.role_prefix}_ECS_Role"
  ec2_profile_name = "${var.role_prefix}_ECS_Role2"
  ecs_profile_name = "${var.role_prefix}_ECS_Role"
  subnet_idx = 3
  ec2_ami = "ami-0d5eff06f840b45e9"
  ecs_ami = "ami-03fe4d5b1d229063a"
  # TODO: Make these to be terraform variables
  ansible_var_path = "../ansible/inventory/group_vars/vars_from_terraform"
  prefect_var_path = "../prefect/vars_from_terraform"
  pvt_key_path = "~/.ssh/tacc_aws"
  pub_key_path = "~/.ssh/tacc_aws.pub"
  dev = "soroosh"
}

###################
provider "aws" { 
  region = "us-east-1"
}


################
data "aws_region" "current" {}

################
data "aws_caller_identity" "current" {}

################
data "aws_availability_zones" "available" {
  state = "available"
}

################
data "aws_s3_object" "odssm-prep-ud" {
  bucket = aws_s3_bucket.odssm-s3["statics"].bucket
  key = "userdata/userdata-ocsmesh.txt"
}

################
data "aws_s3_object" "odssm-solve-ud" {
  bucket = aws_s3_bucket.odssm-s3["statics"].bucket
  key = "userdata/userdata-schism.txt"
}

################
data "aws_s3_object" "odssm-post-ud" {
  bucket = aws_s3_bucket.odssm-s3["statics"].bucket
  key = "userdata/userdata-viz.txt"
}

################
data "aws_s3_object" "odssm-wf-ud" {
  bucket = aws_s3_bucket.odssm-s3["statics"].bucket
  key = "userdata/userdata-wf.txt"
}


###################
resource "local_file" "odssm-ansible-vars" {
    content  = yamlencode({
      ansible_ssh_private_key_file = abspath(pathexpand(local.pvt_key_path))
      aws_account_id = data.aws_caller_identity.current.account_id
      aws_default_region: data.aws_region.current.name
      ec2_public_ip = aws_instance.odssm-local-agent-ec2.public_ip
      ecs_task_role = local.task_role_arn
      ecs_exec_role = local.execution_role_arn
      efs_id = aws_efs_file_system.odssm-efs.id
      prefect_image = aws_ecr_repository.odssm-repo["odssm-workflow"].repository_url
    })
    filename = local.ansible_var_path
}
###################
resource "local_file" "odssm-prefect-vars" {
    content  = yamlencode({
      S3_BUCKET = aws_s3_bucket.odssm-s3["prefect"].bucket
      OCSMESH_CLUSTER = aws_ecs_cluster.odssm-cluster-prep.name
      SCHISM_CLUSTER = aws_ecs_cluster.odssm-cluster-solve.name 
      VIZ_CLUSTER = aws_ecs_cluster.odssm-cluster-post.name
      WF_CLUSTER = aws_ecs_cluster.odssm-cluster-workflow.name
      OCSMESH_TEMPLATE_1_ID = aws_launch_template.odssm-prep-instance-1-template.id
      OCSMESH_TEMPLATE_2_ID = aws_launch_template.odssm-prep-instance-2-template.id
      SCHISM_TEMPLATE_ID = aws_launch_template.odssm-solve-instance-template.id
      VIZ_TEMPLATE_ID = aws_launch_template.odssm-post-instance-template.id
      WF_TEMPLATE_ID = aws_launch_template.odssm-workflow-instance-template.id
      ECS_TASK_ROLE = local.task_role_arn
      ECS_EXEC_ROLE = local.execution_role_arn
      ECS_SUBNET_ID = aws_subnet.odssm-subnet.id
      ECS_EC2_SG = [
        aws_security_group.odssm-sg-efs.id,
        aws_security_group.odssm-sg-ecsout.id,
        aws_security_group.odssm-sg-ssh.id,
    ]
      WF_IMG = "${aws_ecr_repository.odssm-repo["odssm-workflow"].repository_url}:v0.4"
      WF_ECS_TASK_ARN = aws_ecs_task_definition.odssm-flowrun-task.arn
    })
    filename = local.prefect_var_path
}

###################
resource "aws_key_pair" "odssm-ssh-key" {
  key_name = "noaa-ondemand-${local.dev}-tacc-prefect-ssh-key"
  public_key = file("${local.pub_key_path}")
  tags = local.common_tags
}


###################
resource "aws_s3_bucket" "odssm-s3" {
  for_each = {
    statics = "tacc-nos-icogs-static"
    prefect = "tacc-nos-icogs-prefect"
    results = "tacc-icogs-results"
  }
  bucket = "${each.value}"

  tags = merge(
    local.common_tags,
    {
      Reason = "${each.key}"
    }
  )
}


###################
resource "aws_s3_bucket_acl" "odssm-s3-acl" {
  for_each = aws_s3_bucket.odssm-s3
  bucket = each.value.id
  acl = "private"
}


###################
resource "aws_s3_bucket_public_access_block" "odssm-s3-accessblock" {
  for_each = aws_s3_bucket.odssm-s3
  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


###################
resource "aws_s3_bucket" "odssm-s3-website" {
  bucket = "tacc-icogs-results-website"

  tags = merge(
    local.common_tags,
    {
      Reason = "website"
    }
  )
}


###################
resource "aws_s3_bucket_acl" "odssm-s3-website-acl" {
  bucket = aws_s3_bucket.odssm-s3-website.id
  acl    = "public-read"
}

###################
resource "aws_s3_bucket_website_configuration" "odssm-s3-website-config" {
  bucket = aws_s3_bucket.odssm-s3-website.bucket

  index_document {
    suffix = "index.html"
  }

}

###################
resource "aws_efs_file_system" "odssm-efs" {
  tags = local.common_tags
}


###################
resource "aws_efs_mount_target" "odssm-efs-mount" {
  file_system_id = aws_efs_file_system.odssm-efs.id
  subnet_id      = aws_subnet.odssm-subnet.id
  security_groups = [
    aws_security_group.odssm-sg-efs.id
  ] 
}


###################
resource "aws_vpc" "odssm-vpc" {
  assign_generated_ipv6_cidr_block = false
  cidr_block = "172.31.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true
  tags = local.common_tags
}


###################
resource "aws_subnet" "odssm-subnet" {
  vpc_id     = aws_vpc.odssm-vpc.id

  assign_ipv6_address_on_creation = false

  availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]

  cidr_block = "172.31.0.0/20"

  tags = local.common_tags
}


###################
resource "aws_security_group" "odssm-sg-default" {
  name        = "default"
  description = "default VPC security group"
  vpc_id      = aws_vpc.odssm-vpc.id

  egress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 0
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "-1"
      security_groups = []
      self = false
      to_port = 0
    }
  ]
  ingress = [
    {
      cidr_blocks = []
      description = ""
      from_port = 0
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "-1"
      security_groups = []
      self = true
      to_port = 0
    }
  ]

  tags = local.common_tags
}


###################
resource "aws_security_group" "odssm-sg-ecsout" {
  name        = "ecs"
  description = "Allow ecs to access the internet"
  vpc_id      = aws_vpc.odssm-vpc.id

  egress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 0
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "-1"
      security_groups = []
      self = false
      to_port = 0
    }
  ]
  ingress = []

  tags = local.common_tags
}


###################
resource "aws_security_group" "odssm-sg-efs" {
  name        = "efs"
  description = "Allow EFS/NFS mounts"
  vpc_id      = aws_vpc.odssm-vpc.id

  egress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 2049
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "tcp"
      security_groups = []
      self = false
      to_port = 2049
    }
  ]
  ingress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 2049
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "tcp"
      security_groups = []
      self = false
      to_port = 2049
    }
  ]

  tags = local.common_tags
}

###################
resource "aws_security_group" "odssm-sg-ssh" {
  name        = "ssh-access"
  description = "Allow SSH"
  vpc_id      = aws_vpc.odssm-vpc.id

  egress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 22
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "tcp"
      security_groups = []
      self = false
      to_port = 22
    }
  ]
  ingress = [
    {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = ""
      from_port = 22
      ipv6_cidr_blocks = []
      prefix_list_ids = []
      protocol = "tcp"
      security_groups = []
      self = false
      to_port = 22
    }
  ]

  tags = local.common_tags
}


###################
resource "aws_ecr_repository" "odssm-repo" {
  for_each = {
    odssm-info = "Fetch hurricane information"
    odssm-mesh = "Mesh the domain"
    odssm-prep = "Setup SCHISM model"
    odssm-solve = "Run SCHISM model"
    odssm-post = "Generate visualizations"
    odssm-workflow = "Run SCHISM model"
  }
  name = "${each.key}"

  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }


  tags = merge(
    local.common_tags,
    {
      Description = "${each.value}"
    }
  )
}

###################
resource "aws_ecs_cluster" "odssm-cluster-prep" {

  name = "odssm-ocsmesh"

  setting {
    name = "containerInsights"
    value = "disabled"
  }

  tags = merge(
    local.common_tags,
    {
      Description = "Cluster used for model preparation"
    }
  )
}


###################
resource "aws_ecs_cluster" "odssm-cluster-solve" {
  name = "odssm-schism"

  setting {
    name = "containerInsights"
    value = "disabled"
  }

  tags = merge(
    local.common_tags,
    {
      Description = "Cluster used for solving the model"
    }
  )
}

###################
resource "aws_ecs_cluster" "odssm-cluster-post" {
  name = "odssm-viz"

  setting {
    name = "containerInsights"
    value = "disabled"
  }

  tags = merge(
    local.common_tags,
    {
      Description = "Cluster used for generating visualizations"
    }
  )
}


###################
resource "aws_ecs_cluster" "odssm-cluster-workflow" {
  name = "odssm-wf"

  setting {
    name = "containerInsights"
    value = "disabled"
  }

  tags = merge(
    local.common_tags,
    {
      Description = "Cluster used for running Prefect Flows on ECS"
    }
  )
}


###################
resource "aws_ecs_task_definition" "odssm-info-task" {

    family = "odssm-info"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    task_role_arn = local.task_role_arn
    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "info"
        image = "${aws_ecr_repository.odssm-repo["odssm-info"].repository_url}:v0.10"

        essential = true

        memoryReservation = 2000 # MB
        mountPoints = [
          {
            containerPath = "/home/${local.docker_user}/app/io/output"
            sourceVolume = "efs_vol"
          }
        ]
        logConfiguration = {
          logDriver = "awslogs",
          options = {
            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
            awslogs-create-group = "true",
            awslogs-region = data.aws_region.current.name,
            awslogs-stream-prefix = "odssm-info"
          }
        }
      }])

    volume {
      name = "efs_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/hurricanes"
      }
    }

    tags = local.common_tags
}


###################
resource "aws_ecs_task_definition" "odssm-mesh-task" {

    family = "odssm-mesh"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    task_role_arn = local.task_role_arn
    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "mesh"
        image = "${aws_ecr_repository.odssm-repo["odssm-mesh"].repository_url}:v0.11"

        essential = true

        memoryReservation = 123000 # MB
        mountPoints = [
          {
            containerPath = "/home/${local.docker_user}/app/io"
            sourceVolume = "efs_vol"
          },
        ]
        logConfiguration = {
          logDriver = "awslogs",
          options = {
            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
            awslogs-create-group = "true",
            awslogs-region = data.aws_region.current.name,
            awslogs-stream-prefix = "odssm-mesh"
          }
        }
      }])

    volume {
      name = "efs_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/"
      }
    }

    tags = local.common_tags
}


###################
resource "aws_ecs_task_definition" "odssm-prep-task" {

    family = "odssm-prep"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    task_role_arn = local.task_role_arn
    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "prep"
        image = "${aws_ecr_repository.odssm-repo["odssm-prep"].repository_url}:v0.16"

        essential = true

        memoryReservation = 2000 # MB
        mountPoints = [
          {
            containerPath = "/home/${local.docker_user}/app/io/"
            sourceVolume = "efs_vol"
          }
        ]
        logConfiguration = {
          logDriver = "awslogs",
          options = {
            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
            awslogs-create-group = "true",
            awslogs-region = data.aws_region.current.name,
            awslogs-stream-prefix = "odssm-prep"
          }
        }
      }])

    volume {
      name = "efs_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/"
      }
    }

    tags = local.common_tags
}


###################
resource "aws_ecs_task_definition" "odssm-solve-task" {

    family = "odssm-solve"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    task_role_arn = local.task_role_arn
    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "solve"
        image = "${aws_ecr_repository.odssm-repo["odssm-solve"].repository_url}:v0.9"

        essential = true

        environment = [
          {
            name = "SCHISM_NPROCS"
            value = "48"
          }
        ]

        linuxParameters = {
          capabilities = {
            add = ["SYS_PTRACE"]
          }
        }

        memoryReservation = 50000 # MB
        mountPoints = [
          {
            containerPath = "/home/${local.docker_user}/app/io/hurricanes"
            sourceVolume = "hurr_vol"
          }
        ]
        logConfiguration = {
          logDriver = "awslogs",
          options = {
            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
            awslogs-create-group = "true",
            awslogs-region = data.aws_region.current.name,
            awslogs-stream-prefix = "odssm-solve"
          }
        }
      }])

    volume {
      name = "hurr_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/hurricanes"
      }
    }

    tags = local.common_tags
}


###################
resource "aws_ecs_task_definition" "odssm-post-task" {

    family = "odssm-post"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    task_role_arn = local.task_role_arn
    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "post"
        image = "${aws_ecr_repository.odssm-repo["odssm-post"].repository_url}:v0.7"

        essential = true

        memoryReservation = 6000 # MB
        mountPoints = [
          {
            containerPath = "/home/${local.docker_user}/app/io/hurricanes"
            sourceVolume = "hurr_vol"
          }
        ]
        logConfiguration = {
          logDriver = "awslogs",
          options = {
            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
            awslogs-create-group = "true",
            awslogs-region = data.aws_region.current.name,
            awslogs-stream-prefix = "odssm-post"
          }
        }
      }])

    volume {
      name = "hurr_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/hurricanes"
      }
    }

    tags = local.common_tags
}


###################
resource "aws_ecs_task_definition" "odssm-flowrun-task" {

    family = "odssm-prefect-flowrun"
    network_mode = "bridge"
    requires_compatibilities = [ "EC2" ]
    # Use the instance profile the task si running on instead
#    task_role_arn = local.task_role_arn
#    execution_role_arn = local.execution_role_arn

    container_definitions = jsonencode([
      {
        name = "flow"
        image = "${aws_ecr_repository.odssm-repo["odssm-workflow"].repository_url}:v0.4"

        essential = true

        memoryReservation = 500 # MB
        mountPoints = [
          {
            containerPath = "/efs"
            sourceVolume = "efs_vol"
          }
        ]
#        logConfiguration = {
#          logDriver = "awslogs",
#          options = {
#            awslogs-group = aws_cloudwatch_log_group.odssm-cw-log-grp.name,
#            awslogs-create-group = "true",
#            awslogs-region = data.aws_region.current.name,
#            awslogs-stream-prefix = "odssm-prefect-flowrun"
#          }
#        }
      }])

    volume {
      name = "efs_vol"
      efs_volume_configuration {
        file_system_id = aws_efs_file_system.odssm-efs.id
        root_directory = "/"
      }
    }

    tags = local.common_tags
}

###################
resource "aws_instance" "odssm-local-agent-ec2" {

  ami = local.ec2_ami

  associate_public_ip_address = true

  availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]

  iam_instance_profile = local.ec2_profile_name

  instance_type = "t3.small" # micro cannot handle multiple workflow runs, t2 network is slow

  key_name = aws_key_pair.odssm-ssh-key.id

  subnet_id = aws_subnet.odssm-subnet.id

  vpc_security_group_ids = [
#    aws_security_group.odssm-sg-default.id,
    aws_security_group.odssm-sg-efs.id,
    aws_security_group.odssm-sg-ecsout.id,
    aws_security_group.odssm-sg-ssh.id,
  ]

  tags = merge(
    local.common_tags,
    {
      Role = "Workflow management agent"
    }
  )

}


###################
resource "aws_launch_template" "odssm-prep-instance-1-template" {

  name = "odssm-ocsmesh-awsall"
  description = "Instance with sufficient memory for meshing process"
  update_default_version = true

  image_id = local.ecs_ami

  key_name = aws_key_pair.odssm-ssh-key.key_name

  instance_type = "m5.8xlarge"

  iam_instance_profile {
    name = local.ecs_profile_name
  }

  network_interfaces {
    associate_public_ip_address = true
    subnet_id = aws_subnet.odssm-subnet.id
    security_groups = [
#      aws_security_group.odssm-sg-default.id,
      aws_security_group.odssm-sg-efs.id,
      aws_security_group.odssm-sg-ecsout.id,
      aws_security_group.odssm-sg-ssh.id,
    ]
  }


  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size = 300
    }
  }

  ebs_optimized = true

  placement {
    availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      local.common_tags,
      {
        Role = "Run model preparation tasks"
      }
    )
  }

  user_data = base64encode(data.aws_s3_object.odssm-prep-ud.body)
}


###################
resource "aws_launch_template" "odssm-prep-instance-2-template" {

  name = "odssm-ocsmesh-hybrid"
  description = "Instance for pre processing in when meshing is done on HPC"
  update_default_version = true

  image_id = local.ecs_ami

  key_name = aws_key_pair.odssm-ssh-key.key_name

  instance_type = "m5.xlarge"

  iam_instance_profile {
    name = local.ecs_profile_name
  }

  network_interfaces {
    associate_public_ip_address = true
    subnet_id = aws_subnet.odssm-subnet.id
    security_groups = [
#      aws_security_group.odssm-sg-default.id,
      aws_security_group.odssm-sg-efs.id,
      aws_security_group.odssm-sg-ecsout.id,
      aws_security_group.odssm-sg-ssh.id,
    ]
  }


  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size = 300
    }
  }

  ebs_optimized = true

  placement {
    availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      local.common_tags,
      {
        Role = "Run model preparation tasks"
      }
    )
  }

  user_data = base64encode(data.aws_s3_object.odssm-prep-ud.body)
}


###################
resource "aws_launch_template" "odssm-solve-instance-template" {

  name = "odssm-schism"
  description = "Instance with sufficient compute power for SCHISM"
  update_default_version = true

  image_id = local.ecs_ami

  key_name = aws_key_pair.odssm-ssh-key.key_name

  instance_type = "c5.metal"

  iam_instance_profile {
    name = local.ecs_profile_name
  }

  network_interfaces {
    associate_public_ip_address = true
    subnet_id = aws_subnet.odssm-subnet.id
    security_groups = [
#      aws_security_group.odssm-sg-default.id,
      aws_security_group.odssm-sg-efs.id,
      aws_security_group.odssm-sg-ecsout.id,
      aws_security_group.odssm-sg-ssh.id,
    ]
  }

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size = 30
    }
  }

  ebs_optimized = true

  # For ensemble runs where we need many instances, we need to spread
  # over multiple az
#  placement {
#    availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]
#  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      local.common_tags,
      {
        Role = "Run model preparation tasks"
      }
    )
  }

  user_data = base64encode(data.aws_s3_object.odssm-solve-ud.body)
}


###################
resource "aws_launch_template" "odssm-post-instance-template" {

  name = "odssm-viz"
  description = "Instance for generating visualization"
  update_default_version = true

  image_id = local.ecs_ami

  key_name = aws_key_pair.odssm-ssh-key.key_name

  instance_type = "c5.xlarge"

  iam_instance_profile {
    name = local.ecs_profile_name
  }

  network_interfaces {
    associate_public_ip_address = true
    subnet_id = aws_subnet.odssm-subnet.id
    security_groups = [
#      aws_security_group.odssm-sg-default.id,
      aws_security_group.odssm-sg-efs.id,
      aws_security_group.odssm-sg-ecsout.id,
      aws_security_group.odssm-sg-ssh.id,
    ]
  }

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size = 30
    }
  }

  ebs_optimized = true

  placement {
    availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      local.common_tags,
      {
        Role = "Run visualization generation tasks"
      }
    )
  }

  user_data = base64encode(data.aws_s3_object.odssm-post-ud.body)
}

###################
resource "aws_launch_template" "odssm-workflow-instance-template" {

  name = "odssm-wf"
  description = "Instance for running Prefect Flows as ECSRun"
  update_default_version = true

  image_id = local.ecs_ami

  key_name = aws_key_pair.odssm-ssh-key.key_name

  instance_type = "c5.4xlarge"

  # The workflow ECSRun instance needs to be able to create its own instances
  iam_instance_profile {
    name = local.ec2_profile_name
  }

  network_interfaces {
    associate_public_ip_address = true
    subnet_id = aws_subnet.odssm-subnet.id
    security_groups = [
#      aws_security_group.odssm-sg-default.id,
      aws_security_group.odssm-sg-efs.id,
      aws_security_group.odssm-sg-ecsout.id,
      aws_security_group.odssm-sg-ssh.id,
    ]
  }

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size = 30
    }
  }

  ebs_optimized = true

  placement {
    availability_zone = data.aws_availability_zones.available.names[local.subnet_idx]
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(
      local.common_tags,
      {
        Role = "Run ECSRun tasks"
      }
    )
  }

  user_data = base64encode(data.aws_s3_object.odssm-wf-ud.body)
}

resource "aws_cloudwatch_log_group" "odssm-cw-log-grp" {
  name = "odssm_ecs_task_docker_logs"

#  kms_key_id = 

  tags = merge(
    local.common_tags,
    {
      Role = "Watch ECS logs"
    }
  )
}
