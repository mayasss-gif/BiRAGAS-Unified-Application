import re
import logging

from .label_info import get_drug_label_info
from .name_and_route import get_drug_names_and_route
from .adverse_reactions import get_adverse_reactions
from .query_fda import query_openfda
from .models import (
    FDAApprovalInfo, 
    DrugLabelInfo, 
    DrugNamesAndRoute,
    create_approved_drug,
    create_not_found_drug,
    create_error_drug
)

logger = logging.getLogger(__name__)

def verify_fda_approval(drug_name: str) -> FDAApprovalInfo:
    """
    Verify FDA approval status for a drug using OpenFDA API.

    Args:
        drug_name: Name of the drug to verify

    Returns:
        FDAApprovalInfo object with FDA approval information
    """
    try:
        # Clean drug name
        clean_name = re.sub(r"\s*\(.*?\)", "", drug_name).strip().lower()
        logger.debug(f"Checking FDA approval for: {clean_name}")
        
        # Check if drug is approved using query_openfda
        drug_fda_info = query_openfda(clean_name, field_type="generic")

        if drug_fda_info is None:
            drug_fda_info = query_openfda(clean_name, field_type="brand")

        if drug_fda_info:
            # Handle case where products might be a list
            products = drug_fda_info.get('products', {})
            if isinstance(products, list) and len(products) > 0:
                products = products[0]
            elif not isinstance(products, dict):
                products = {}
            
            # Handle case where openfda might be a list
            openfda = drug_fda_info.get('openfda', {})
            if isinstance(openfda, list) and len(openfda) > 0:
                openfda = openfda[0]
            elif not isinstance(openfda, dict):
                openfda = {}
            
            # Handle generic_name which might be a list
            generic_name = openfda.get('generic_name', 'Not available')
            if isinstance(generic_name, list) and len(generic_name) > 0:
                generic_name = generic_name[0]
            elif not isinstance(generic_name, str):
                generic_name = 'Not available'
            
            return create_approved_drug(
                drug_name=drug_name,
                brand_name=products.get('brand_name', 'Not available'),
                generic_name=generic_name,
                route=products.get('route', 'Not available'),
                indications=products.get('indication', 'Not available')
            )

        # Get drug label info
        label_info_dict = get_drug_label_info(clean_name)
        label_info = DrugLabelInfo.from_dict(label_info_dict) if label_info_dict else None

        # Get drug names and route of administration
        ndc_info_dict = get_drug_names_and_route(clean_name)
        ndc_info = DrugNamesAndRoute.from_dict(ndc_info_dict) if ndc_info_dict else None

        # Get adverse reactions
        reactions = get_adverse_reactions(clean_name)

        if label_info_dict and ndc_info_dict:
            return create_approved_drug(
                drug_name=drug_name,
                brand_name=ndc_info_dict.get('Brand Name', 'Not available'),
                generic_name=ndc_info_dict.get('Generic Name', 'Not available'),
                route=ndc_info_dict.get('Route of Administration', 'Not available'),
                indications=label_info_dict.get('Indications and Usage', 'Not available'),
                reactions=reactions,
                label_info=label_info
            )
        else:
            return create_not_found_drug(drug_name)

    except Exception as e:
        logger.warning(f"Error checking FDA approval for {drug_name}: {str(e)}")
        return create_error_drug(drug_name, str(e))


def verify_fda_approval_dict(drug_name: str) -> dict:
    """
    Legacy function that returns dictionary format for backward compatibility.
    
    Args:
        drug_name: Name of the drug to verify
        
    Returns:
        Dictionary with FDA approval information (legacy format)
    """
    fda_info = verify_fda_approval(drug_name)
    return fda_info.to_dict()





