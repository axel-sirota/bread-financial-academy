"""
Lambda to delete ALL SageMaker Studio apps on Saturday 12:00 AM.
Guarantees $0 cost Saturday through Thursday.
Students recreate apps automatically when they log in next Friday.
"""

import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """Delete all SageMaker Studio apps for the academy domain."""

    sm = boto3.client('sagemaker')
    domain_name = os.environ.get('DOMAIN_NAME', 'bread-financial-academy')

    deleted_apps = []
    errors = []

    # Find the domain
    domains = sm.list_domains()['Domains']
    domain = next((d for d in domains if d['DomainName'] == domain_name), None)

    if not domain:
        logger.warning(f"Domain {domain_name} not found")
        return {'statusCode': 404, 'message': f'Domain {domain_name} not found'}

    domain_id = domain['DomainId']
    logger.info(f"Found domain: {domain_name} ({domain_id})")

    # List all user profiles
    user_profiles = []
    paginator = sm.get_paginator('list_user_profiles')
    for page in paginator.paginate(DomainIdEquals=domain_id):
        user_profiles.extend(page['UserProfiles'])

    logger.info(f"Found {len(user_profiles)} user profiles")

    # For each user profile, delete all apps
    for profile in user_profiles:
        profile_name = profile['UserProfileName']

        # List apps for this user
        try:
            apps = sm.list_apps(
                DomainIdEquals=domain_id,
                UserProfileNameEquals=profile_name
            )['Apps']
        except Exception as e:
            logger.error(f"Error listing apps for {profile_name}: {e}")
            errors.append(f"list_apps({profile_name}): {e}")
            continue

        for app in apps:
            app_name = app['AppName']
            app_type = app['AppType']
            status = app.get('Status', 'Unknown')

            # Skip if already deleted/deleting
            if status in ['Deleted', 'Deleting']:
                continue

            try:
                logger.info(f"Deleting {app_type} app '{app_name}' for {profile_name}")
                sm.delete_app(
                    DomainId=domain_id,
                    UserProfileName=profile_name,
                    AppType=app_type,
                    AppName=app_name
                )
                deleted_apps.append(f"{profile_name}/{app_type}/{app_name}")
            except Exception as e:
                logger.error(f"Error deleting app {app_name}: {e}")
                errors.append(f"delete_app({profile_name}/{app_name}): {e}")

    # ========================================================================
    # SHARED SPACE APP CLEANUP
    # Delete JupyterLab apps in shared spaces (e.g., shared-academy-workspace)
    # ========================================================================
    logger.info("Checking for apps in shared spaces...")

    try:
        # List shared spaces in the domain
        spaces = sm.list_spaces(
            DomainIdEquals=domain_id,
            SpaceTypeEquals='Shared'
        )['Spaces']

        logger.info(f"Found {len(spaces)} shared spaces")

        for space in spaces:
            space_name = space['SpaceName']

            # List apps in this shared space
            try:
                apps = sm.list_apps(
                    DomainIdEquals=domain_id,
                    SpaceNameEquals=space_name
                )['Apps']

                logger.info(f"Shared space '{space_name}' has {len(apps)} apps")

                for app in apps:
                    app_name = app['AppName']
                    app_type = app['AppType']
                    status = app.get('Status', 'Unknown')

                    # Skip if already deleted/deleting
                    if status in ['Deleted', 'Deleting']:
                        continue

                    try:
                        logger.info(f"Deleting {app_type} app '{app_name}' in shared space '{space_name}'")
                        sm.delete_app(
                            DomainId=domain_id,
                            SpaceName=space_name,
                            AppType=app_type,
                            AppName=app_name
                        )
                        deleted_apps.append(f"shared:{space_name}/{app_type}/{app_name}")
                    except Exception as e:
                        logger.error(f"Error deleting shared space app {app_name}: {e}")
                        errors.append(f"delete_app(shared:{space_name}/{app_name}): {e}")

            except Exception as e:
                logger.error(f"Error listing apps for shared space {space_name}: {e}")
                errors.append(f"list_apps(shared:{space_name}): {e}")

    except Exception as e:
        logger.error(f"Error listing shared spaces: {e}")
        errors.append(f"list_spaces: {e}")

    result = {
        'statusCode': 200,
        'domain': domain_name,
        'user_profiles_checked': len(user_profiles),
        'apps_deleted': len(deleted_apps),
        'deleted_apps': deleted_apps,
        'errors': errors
    }

    logger.info(f"Cleanup complete: {len(deleted_apps)} apps deleted, {len(errors)} errors")
    return result
