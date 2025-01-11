import os
import os.path as osp
from pathlib import Path
import json
import enum
import sys
import argparse
import abc
import datetime
import random
import gc

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Table, MetaData, ForeignKey
from sqlalchemy import Integer, String, Date, DateTime, Float, Boolean, Text, Enum
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column
from tqdm import tqdm


Base = declarative_base()
__db_engine = None


def get_db_engine():
    """
    Return a globally cached db engine (from DB_CONNECTION_STRING envvar) after creating one if it does not already exist.
    """
    global __db_engine
    if __db_engine is not None:
        return __db_engine
    else:
        print("Global db engine has not been initialized. Initializing...")
        __db_engine = db_init()
        return __db_engine


def db_init(conn_str=None):
    """
    Instantiate and return a new database engine backed by SQLAlchemy.
    If conn_str is not provided, it will use the DB_CONNECTION_STRING environment variable instead.
    """
    if conn_str is None:
        conn_str = os.getenv('DB_CONNECTION_STRING', None)
    if not conn_str:
        raise Exception("'DB_CONNECTION_STRING' environment variable not found.")
    else:
        db_engine = create_engine(conn_str)
        Base.metadata.create_all(db_engine)
        print("Initialized database engine")
        return db_engine


class EType(enum.Enum):
    Training = 'Training'
    Holdout = 'Holdout'


def get_etype(s):
    return {'Training': EType.Training, 'Holdout': EType.Holdout}[s]


class Member(Base):
    __tablename__ = 'member'

    member_id = Column(Integer, primary_key=True)
    data_type = Column(Enum(EType))


class EProductType(enum.Enum):
    LPPO = 'LPPO'


class EPlanCategory(enum.Enum):
    MedicareAdvantage = 'Medicare Advantage'


class RawTargetMembers(Base):
    __tablename__ = 'raw_target_members'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    calendar_year = Column(Integer)
    product_type = Column(Enum(EProductType))
    plan_category = Column(Enum(EPlanCategory))
    preventive_visit_gap_ind = Column(Boolean)


class RawAdditionalFeaturesRaw(Base):
    __tablename__ = 'raw_additional_features'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))

    cci_score = Column(Float)
    dcsi_score = Column(Integer)
    fci_score = Column(Integer)
    cms_tot_partd_payment_amt = Column(Float)
    cms_tot_ma_payment_amt = Column(Float)
    cms_frailty_ind = Column(Boolean)
    atlas_grocpth14 = Column(Float)
    atlas_povertyallagespct = Column(Float)
    atlas_recfacpth14 = Column(Float)
    atlas_ffrpth14 = Column(Float)
    atlas_fsrpth14 = Column(Float)


class RawMarketingControlPoint(Base):
    __tablename__ = 'raw_marketing_control_point'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))

    cnt_cp_emails_0 = Column(Integer)
    cnt_cp_emails_1 = Column(Integer)
    cnt_cp_emails_10 = Column(Integer)
    cnt_cp_emails_11 = Column(Integer)
    cnt_cp_emails_2 = Column(Integer)
    cnt_cp_emails_3 = Column(Integer)
    cnt_cp_emails_4 = Column(Integer)
    cnt_cp_emails_5 = Column(Integer)
    cnt_cp_emails_6 = Column(Integer)
    cnt_cp_emails_7 = Column(Integer)
    cnt_cp_emails_8 = Column(Integer)
    cnt_cp_emails_9 = Column(Integer)
    cnt_cp_emails_pmpm_ct = Column(Float)
    cnt_cp_livecall_0 = Column(Integer)
    cnt_cp_livecall_1 = Column(Integer)
    cnt_cp_livecall_10 = Column(Integer)
    cnt_cp_livecall_11 = Column(Integer)
    cnt_cp_livecall_2 = Column(Integer)
    cnt_cp_livecall_3 = Column(Integer)
    cnt_cp_livecall_4 = Column(Integer)
    cnt_cp_livecall_5 = Column(Integer)
    cnt_cp_livecall_6 = Column(Integer)
    cnt_cp_livecall_7 = Column(Integer)
    cnt_cp_livecall_8 = Column(Integer)
    cnt_cp_livecall_9 = Column(Integer)
    cnt_cp_livecall_pmpm_ct = Column(Float)
    cnt_cp_print_0 = Column(Integer)
    cnt_cp_print_1 = Column(Integer)
    cnt_cp_print_10 = Column(Integer)
    cnt_cp_print_11 = Column(Integer)
    cnt_cp_print_2 = Column(Integer)
    cnt_cp_print_3 = Column(Integer)
    cnt_cp_print_4 = Column(Integer)
    cnt_cp_print_5 = Column(Integer)
    cnt_cp_print_6 = Column(Integer)
    cnt_cp_print_7 = Column(Integer)
    cnt_cp_print_8 = Column(Integer)
    cnt_cp_print_9 = Column(Integer)
    cnt_cp_print_pmpm_ct = Column(Float)
    cnt_cp_vat_0 = Column(Integer)
    cnt_cp_vat_1 = Column(Integer)
    cnt_cp_vat_10 = Column(Integer)
    cnt_cp_vat_11 = Column(Integer)
    cnt_cp_vat_2 = Column(Integer)
    cnt_cp_vat_3 = Column(Integer)
    cnt_cp_vat_4 = Column(Integer)
    cnt_cp_vat_5 = Column(Integer)
    cnt_cp_vat_6 = Column(Integer)
    cnt_cp_vat_7 = Column(Integer)
    cnt_cp_vat_8 = Column(Integer)
    cnt_cp_vat_9 = Column(Integer)
    cnt_cp_vat_pmpm_ct = Column(Float)
    cnt_cp_webstatement_0 = Column(Integer)
    cnt_cp_webstatement_1 = Column(Integer)
    cnt_cp_webstatement_10 = Column(Integer)
    cnt_cp_webstatement_11 = Column(Integer)
    cnt_cp_webstatement_2 = Column(Integer)
    cnt_cp_webstatement_3 = Column(Integer)
    cnt_cp_webstatement_4 = Column(Integer)
    cnt_cp_webstatement_5 = Column(Integer)
    cnt_cp_webstatement_6 = Column(Integer)
    cnt_cp_webstatement_7 = Column(Integer)
    cnt_cp_webstatement_8 = Column(Integer)
    cnt_cp_webstatement_9 = Column(Integer)
    cnt_cp_webstatement_pmpm_ct = Column(Float)


class RawCostAndUtilization(Base):
    __tablename__ = 'raw_cost_and_utilization'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))

    bh_psyc_visit_ct_pmpm = Column(Float)
    bh_rtc_admit_ct_pmpm = Column(Float)
    bh_rtc_admit_days_pmpm = Column(Float)
    days_since_last_clm = Column(Integer)
    nonpar_allowed_pmpm_cost = Column(Float)
    nonpar_clm_ct_pmpm = Column(Float)
    nonpar_cob_paid_pmpm_cost = Column(Float)
    nonpar_coins_pmpm_cost = Column(Float)
    nonpar_copay_pmpm_cost = Column(Float)
    nonpar_deduct_pmpm_cost = Column(Float)
    nonpar_ds_clm = Column(Integer)
    nonpar_mbr_resp_pmpm_cost = Column(Float)
    nonpar_net_paid_pmpm_cost = Column(Float)
    oontwk_allowed_pmpm_cost = Column(Float)
    oontwk_clm_ct_pmpm = Column(Float)
    oontwk_cob_paid_pmpm_cost = Column(Float)
    oontwk_coins_pmpm_cost = Column(Float)
    oontwk_copay_pmpm_cost = Column(Float)
    oontwk_deduct_pmpm_cost = Column(Float)
    oontwk_ds_clm = Column(Integer)
    oontwk_mbr_resp_pmpm_cost = Column(Float)
    oontwk_net_paid_pmpm_cost = Column(Float)
    total_allowed_pmpm_cost = Column(Float)
    total_cob_paid_pmpm_cost = Column(Float)
    total_coins_pmpm_cost = Column(Float)
    total_copay_pmpm_cost = Column(Float)
    total_deduct_pmpm_cost = Column(Float)
    total_ip_acute_admit_days_pmpm = Column(Float)
    total_ip_ltach_admit_days_pmpm = Column(Float)
    total_ip_maternity_admit_days_pmpm = Column(Float)
    total_ip_mhsa_admit_days_pmpm = Column(Float)
    total_ip_rehab_admit_days_pmpm = Column(Float)
    total_ip_snf_admit_days_pmpm = Column(Float)
    total_mbr_resp_pmpm_cost = Column(Float)
    total_net_paid_pmpm_cost = Column(Float)

class ERUCC(enum.Enum):
    Metro_1 = '1-Metro'
    Metro_2 = '2-Metro'
    Metro_3 = '3-Metro'
    Metro_4 = '4-Nonmetro'
    Nonmetro_5 = '5-Nonmetro'
    Nonmetro_6 = '6-Nonmetro'
    Nonmetro_7 = '7-Nonmetro'
    Nonmetro_8 = '8-Nonmetro'
    Nonemtro_9 = '9-Nonmetro'

class ELangSpoken(enum.Enum):
    ENG = 'ENG'; OTH = 'OTH'; SPA = 'SPA'; CPF = 'CPF'; CHI = 'CHI'; KOR = 'KOR'; FRE = 'FRE'
    VIE = 'VIE'; YUE = 'YUE'; FAS = 'FAS'; PER = 'PER'; JPN = 'JPN'; POL = 'POL'; RUS = 'RUS'
    POR = 'POR'; TGL = 'TGL'; ARA = 'ARA'; ITA = 'ITA'; GER = 'GER'; CRE = 'CRE'; CMN = 'CMN'
    LAO = 'LAO'; DUT = 'DUT'; THA = 'THA'; NAV = 'NAV'; ZZZ = 'ZZZ'; TAG = 'TAG'; HMG = 'HMG';
    BEN = 'BEN'; PHI = 'PHI'; MAN = 'MAN'; HIN = 'HIN'; GUJ = 'GUJ'; SRP = 'SRP'; IRA = 'IRA'
    URD = 'URD'


class RawDemographics(Base):
    __tablename__ = 'raw_demographics'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))

    riskarr_downside = Column(Boolean)
    riskarr_global = Column(Boolean)
    riskarr_rewards = Column(Boolean)
    riskarr_upside = Column(Boolean)
    rucc_category = Column(Enum(ERUCC))
    lang_spoken_cd = Column(Enum(ELangSpoken))


class RawPharmacyUtilization(Base):
    __tablename__ = 'raw_pharmacy_utilization'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))

    rx_days_since_last_script = Column(Integer)
    rx_overall_coins_pmpm_cost = Column(Float)
    rx_overall_copay_pmpm_cost = Column(Float)
    rx_overall_deduct_pmpm_cost = Column(Float)
    rx_overall_dist_gpi6_pmpm_ct = Column(Float)
    rx_overall_gpi_pmpm_ct = Column(Float)
    rx_overall_mbr_resp_pmpm_cost = Column(Float)
    rx_overall_net_paid_pmpm_cost = Column(Float)
    rx_overall_pmpm_cost = Column(Float)
    rx_overall_pmpm_ct = Column(Float)
    rx_perphy_pmpm_ct = Column(Float)
    rx_pharmacies_pmpm_ct = Column(Float)
    rx_tier_1_pmpm_ct = Column(Float)
    rx_tier_2_pmpm_ct = Column(Float)
    rx_tier_3_pmpm_ct = Column(Float)
    rx_tier_4_pmpm_ct = Column(Float)


class EChannel(enum.Enum):
    Field = 'Field'
    ConsumerDirect = 'Consumer Direct'
    DMSTelesales = 'DMS Telesales'
    PartnerCallCenter = 'Partner Call Center'
    Brokerage = 'Brokerage'


class RawChannel(Base):
    __tablename__ = 'raw_channel'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    channel = Column(Enum(EChannel))


class RawSocialDeterminantsOfHealth(Base):
    __tablename__ = 'raw_social_determinants_of_health'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    rwjf_preventable_ip_rate = Column(Float)
    rwjf_healthcare_cost = Column(Float)
    rwjf_other_pcp = Column(Float)
    rwjf_uninsured_adults_pct = Column(Float)
    rwjf_uninsured_child_pct = Column(Float)
    rwjf_diabetes_monitor_pct = Column(Float)
    rwjf_flu_vax = Column(Float)
    rwjf_mammography_pct = Column(Float)
    rwjf_uninsured_pct = Column(Float)
    rwjf_pcp_rate = Column(Float)
    rwjf_dentists_ratio = Column(Float)
    rwjf_men_hlth_prov_ratio = Column(Float)
    rwjf_age_gt_65_pct = Column(Float)
    rwjf_native_race_pct = Column(Float)
    rwjf_asian_race_pct = Column(Float)
    rwjf_age_lt_18_pct = Column(Float)
    rwjf_female_pct = Column(Float)
    rwjf_hispanic_pct = Column(Float)
    rwjf_hawaiian_race_pct = Column(Float)
    rwjf_african_race_pct = Column(Float)
    rwjf_white_race_pct = Column(Float)
    rwjf_non_english_pct = Column(Float)
    rwjf_rural_pct = Column(Float)
    rwjf_population = Column(Float)
    rwjf_drug_overdose_deaths_rate = Column(Float)
    rwjf_drug_deaths_modl_rate = Column(Float)
    rwjf_food_insecurity_pct = Column(Float)
    rwjf_food_env_inx = Column(Float)
    rwjf_insufficient_sleep_pct = Column(Float)
    rwjf_limit_hlthy_food_pct = Column(Float)
    rwjf_mv_deaths_rate = Column(Float)
    rwjf_teen_births_rate = Column(Float)
    rwjf_std_infect_rate = Column(Float)
    rwjf_inactivity_pct = Column(Float)
    rwjf_alcoholic_pct = Column(Float)
    rwjf_adult_obesity_pct = Column(Float)
    rwjf_adult_smoking_pct = Column(Float)
    rwjf_dui_deaths_pct = Column(Float)
    rwjf_exercise_access_pct = Column(Float)
    rwjf_mental_distress_pct = Column(Float)
    rwjf_physical_distress_pct = Column(Float)
    rwjf_premature_death_rate = Column(Float)
    rwjf_poor_men_hlth_days = Column(Float)
    rwjf_poor_phy_hlth_days = Column(Float)
    rwjf_life_expectancy = Column(Float)
    rwjf_child_mortality = Column(Float)
    rwjf_diabetes_pct = Column(Float)
    rwjf_hiv_rate = Column(Float)
    rwjf_infant_mortality = Column(Float)
    rwjf_poor_health_pct = Column(Float)
    rwjf_low_birthweight_pct = Column(Float)
    rwjf_premature_mortality = Column(Float)
    rwjf_long_commute_alone_pct = Column(Float)
    rwjf_air_pollute_density = Column(Float)
    rwjf_drinkwater_violate_ind = Column(Boolean)
    rwjf_housing_cost_burden_pct = Column(Float)
    rwjf_severe_housing_pct = Column(Float)
    rwjf_broadband_access = Column(Float)
    rwjf_home_ownership_pct = Column(Float)
    rwjf_drive_alone_pct = Column(Float)
    rwjf_disconnect_youth_pct = Column(Float)
    rwjf_child_free_lunch_pct = Column(Float)
    rwjf_firearm_fatalities_rate = Column(Float)
    rwjf_homicides_rate = Column(Float)
    rwjf_median_house_income = Column(Float)
    rwjf_injury_deaths_rate = Column(Float)
    rwjf_social_associate_rate = Column(Float)
    rwjf_violent_crime_rate = Column(Float)
    rwjf_some_college_pct = Column(Float)
    rwjf_single_parent_pct = Column(Float)
    rwjf_child_poverty_pct = Column(Float)
    rwjf_high_school_pct = Column(Float)
    rwjf_unemploy_pct = Column(Float)
    rwjf_income_inequ_ratio = Column(Float)
    rwjf_resident_seg_black_inx = Column(Float)
    rwjf_resident_seg_nonwhite_inx = Column(Float)
    rwjf_suicides_rate = Column(Float)


class RawWebActivity(Base):
    __tablename__ = 'raw_web_activity'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    login_count_0 = Column(Integer)
    login_count_1 = Column(Integer)
    login_count_10 = Column(Integer)
    login_count_11 = Column(Integer)
    login_count_2 = Column(Integer)
    login_count_3 = Column(Integer)
    login_count_4 = Column(Integer)
    login_count_5 = Column(Integer)
    login_count_6 = Column(Integer)
    login_count_7 = Column(Integer)
    login_count_8 = Column(Integer)
    login_count_9 = Column(Integer)
    login_pmpm_ct = Column(Float)
    days_since_last_login = Column(Integer)

class ETenureBand(enum.Enum):
    _0_0p5 = '0 - 0.5 YEARS'
    _0p5_1 = '0.5 - 1 YEARS'
    _1_1p5 = '1 - 1.5 YEARS'
    _1p5_2 = '1.5 - 2 YEARS'
    _2_3 = '2 - 3 YEARS'
    _3_4 = '3 - 4 YEARS'
    _4_5 = '4 - 5 YEARS'
    _5_6 = '5 - 6 YEARS'
    _6_7 = '6 - 7 YEARS'
    _7plus = '7+ YEARS'


class RawMemberData(Base):
    __tablename__ = 'raw_member_data'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    consec_tenure_month = Column(Integer)
    all_mm_tenure = Column(Integer)
    tenure_band = Column(Enum(ETenureBand))
    dual_eligible_ind = Column(Boolean)
    disabled_ind = Column(Boolean)
    lis_ind = Column(Boolean)


class EMeasureName(enum.Enum):
    ADH_DIAB = 'ADH (DIAB)'; ADH_ACE = 'ADH (ACE)'; ABA = 'ABA'; ADH_STATIN = 'ADH (STATIN)'
    PCR = 'PCR'; TRC_MRP = 'TRC (MRP)'; CBP = 'CBP'; COL = 'COL'; CDC_EYE = 'CDC (EYE)'
    CDC_NPH = 'CDC (NPH)'; CHC_HbA1c = 'CDC (HbA1c)'; BCS = 'BCS'; SUPD = 'SUPD'
    SPC_STATIN = 'SPC STATIN'; ART = 'ART'; OMW = 'OMW'; COA_MDR = 'COA (MDR)'
    COA_PNS = 'COA (PNS)'; COA_FSA = 'COA (FSA)'; COL_45_50 = 'COL (45-50)'; EED = 'EED'
    FMC = 'FMC'; HBD = 'HBD'; KED = 'KED'; MRP = 'MRP'; TRC_PED = 'TRC (PED)'; ESA = 'ESA'
    SWT = 'SWT'; EGR = 'EGR'; RXC = 'RXC'; ETA = 'ETA'; MDR = 'MDR'; ASV = 'ASV'; TFP = 'TFP'
    SFT = 'SFT'; TBC = 'TBC'; SBT = 'SBT'; TEX = 'TEX'


class EMeasureType(enum.Enum):
    PatientSafety = 'Patient Safety'
    HEDIS = 'HEDIS'
    PatientExperience = 'Patient Experience'


class RawQualityData(Base):
    __tablename__ = 'raw_quality_data'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    measurement_year = Column(Integer)
    measure_name = Column(Enum(EMeasureName))
    measure_desc = Column(Text)
    measure_type = Column(Enum(EMeasureType))
    base_event_date = Column(Date)
    compliant_cnt = Column(Float)
    eligible_cnt = Column(Integer)

class ESex(enum.Enum):
    F = 'F'
    M = 'M'
    U = 'U'

class EMcoContractNbr(enum.Enum):
    H5216 = 'H5216'; H8087 = 'H8087'; H5970 = 'H5970'; H5525 = 'H5525'; H9070 = 'H9070'
    H7284 = 'H7284'; H0473 = 'H0473'; H7617 = 'H7617'; H2029 = 'H2029'; H6622 = 'H6622'; H1036 = 'H1036'

class EState(enum.Enum):
    FL = 'FL'; MS = 'MS'; TX = 'TX'; AL = 'AL'; TN = 'TN'; NC = 'NC'; KY = 'KY'; VA = 'VA'
    MI = 'MI'; NY = 'NY'; GA = 'GA'; NJ = 'NJ'; WA = 'WA'; WI = 'WI'; WV = 'WV'; KS = 'KS'
    MN = 'MN'; IL = 'IL'; HI = 'HI'; SC = 'SC'; NM = 'NM'; AR = 'AR'; OH = 'OH'; OK = 'OK'
    MA = 'MA'; OR = 'OR'; MT = 'MT'; PA = 'PA'; UT = 'UT'; IA = 'IA'; CO = 'CO'; ME = 'ME'
    LA = 'LA'; AZ = 'AZ'; IN = 'IN'; NV = 'NV'; ID = 'ID'; SD = 'SD'; ND = 'ND'; MO = 'MO'
    MD = 'MD'; NH = 'NH'; AK = 'AK'; NE = 'NE'; CA = 'CA'; PR = 'PR'; CT = 'CT'; WY = 'WY'
    VT = 'VT'; RI = 'RI'; DE = 'DE'; DC = 'DC'; VI = 'VI'; GU = 'GU'; MP = 'MP'; AS = 'AS'

class ERace(enum.Enum):
    Black = 'BLACK'
    NAmericanNative = 'N AMERICAN NATIVE'
    White = 'WHITE'
    Asian = 'ASIAN'
    Hispanice = 'HISPANIC'
    Unknown = 'UNKNOWN'
    Other = 'OTHER'


class RawMemberDetails(Base):
    __tablename__ = 'raw_member_details'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    generic_grouper = Column(Boolean)
    unattributed_provider = Column(Boolean)
    sex_cd = Column(Enum(ESex))
    age = Column(Integer)
    veteran_ind = Column(Boolean)
    mco_contract_nbr = Column(Enum(EMcoContractNbr))
    plan_benefit_package_id = Column(Integer)
    state_of_residence = Column(Enum(EState))
    county_of_residence = Column(Text)
    race = Column(Enum(ERace))


class RawMemberClaims(Base):
    __tablename__ = 'raw_member_claims'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    dos_year = Column(Integer)
    clm_unique_key = Column(Text)
    serv_date_skey = Column(Date)
    pcp_visit = Column(Boolean)
    annual_wellness = Column(Boolean)
    humana_paf = Column(Boolean)
    preventative_visit = Column(Boolean)
    comp_physical_exam = Column(Boolean)
    ihwa = Column(Boolean)
    fqhc_visit = Column(Boolean)
    telehealth = Column(Boolean)
    endocrinologist_visit = Column(Boolean)
    oncolologist_visit = Column(Boolean)
    radiologist_visit = Column(Boolean)
    podiatrist_visit = Column(Boolean)
    ophthalmologist_visit = Column(Boolean)
    optometrist_visit = Column(Boolean)
    physical_therapist_visit = Column(Boolean)
    cardiologist_visit = Column(Boolean)
    gastroenterologist_visit = Column(Boolean)
    orthopedist_visit = Column(Boolean)
    obgyn_visit = Column(Boolean)
    nephroloogist_visit = Column(Boolean)
    pulmonologist_visit = Column(Boolean)
    urgent_care_visit = Column(Boolean)
    er_visit = Column(Boolean)


class EChronicity(enum.Enum):
    Chronic = 'Chronic'

class EHccModelType(enum.Enum):
    Medical = 'MEDICAL'
    ESRD = 'ESRD'

class ECMSModelVers(enum.Enum):
    V28 = 'V28'
    V24 = 'V24'

class RawMemberCondition(Base):
    __tablename__ = 'raw_member_condition'

    id = Column(Integer, autoincrement=True, primary_key=True)
    member_id = mapped_column(ForeignKey("member.member_id"))
    cond_key = Column(Integer)
    chronicity = Column(Enum(EChronicity))
    cond_desc = Column(Text)
    hcc_model_type = Column(Enum(EHccModelType))
    cms_model_vers_cd = Column(Enum(ECMSModelVers))
    membership_year = Column(Integer)


class VarHandlerBase(abc.ABC):
    @abc.abstractmethod
    def handle(self, var):
        pass


class FloatHandler(VarHandlerBase):
    def __init__(self):
        super().__init__()

    def handle(self, var):
        return float(var)


class IntegerHandler(VarHandlerBase):
    def __init__(self):
        super().__init__()

    def handle(self, var):
        if pd.isna(var):
            return None
        if pd.api.types.is_numeric_dtype(var):
            try:
                return int(var)
            except:
                return None
        raise ValueError


class BooleanHandler(VarHandlerBase):
    def __init__(self):
        super().__init__()

    def handle(self, var):
        if pd.isna(var):
            return None
        if pd.api.types.is_bool_dtype(var):
            return bool(var)
        if pd.api.types.is_numeric_dtype(var):
            try:
                return bool(int(var))
            except:
                return None

class TextHandler(VarHandlerBase):
    def __init__(self):
        super().__init__()

    def handle(self, var):
        if isinstance(var, str):
            return var
        else:
            return str(var)


class EnumHandler(VarHandlerBase):
    def __init__(self, enum_class):
        super().__init__()
        self.str2enum = {}
        self.indices = {}
        counter = 1
        for fieldname in dir(enum_class):
            field = getattr(enum_class, fieldname)
            if hasattr(field, 'name') and hasattr(field, 'value'):
                strval = getattr(field, 'value')
                self.str2enum[strval] = field
                self.indices[field] = counter
                counter += 1

    def handle(self, var):
        if not isinstance(var, str):
            if pd.isna(var):
                return None
            else:
                raise ValueError
        return self.str2enum[var]
    
    def to_onehot(self, enum_val):
        if isinstance(enum_val, str):
            enum_val = self.str2enum[enum_val]
        a = np.zeros(len(self)+1, dtype=int)
        a[self.indices[enum_val]] = 1
        return a
    
    def __len__(self):
        return len(self.str2enum)


class DateHandler(VarHandlerBase):
    def __init__(self, date_fmt):
        super().__init__()
        self.date_fmt = date_fmt

    def handle(self, var):
        if pd.isna(var):
            return None
        if isinstance(var, datetime.date):
            return var
        elif isinstance(var, datetime.datetime):
            return var.date()
        else:
            if not isinstance(var, str):
                var = str(var)
            return datetime.datetime.strptime(var, self.date_fmt).date()


def get_handlers(dbitem_class, **kw):
    handlers = {}
    for varname in dir(dbitem_class):
        var = getattr(dbitem_class, varname)
        if not hasattr(var, 'type'): continue
        p = kw.get(varname, {})
        if isinstance(var.type, Float):
            handlers[varname] = FloatHandler(**p)
        elif isinstance(var.type, Integer):
            handlers[varname] = IntegerHandler(**p)
        elif isinstance(var.type, Boolean):
            handlers[varname] = BooleanHandler(**p)
        elif isinstance(var.type, Text):
            handlers[varname] = TextHandler(**p)
        elif isinstance(var.type, Enum):
            handlers[varname] = EnumHandler(**{'enum_class': var.type.enum_class, **p})
        elif isinstance(var.type, Date):
            handlers[varname] = DateHandler(**p)
    return handlers


def get_db_class(name):
    dd = {'humana_mays_target_members': RawTargetMembers,
        'Additional Features': RawAdditionalFeaturesRaw,
        'Control Point': RawMarketingControlPoint,
        'Cost & Utilization': RawCostAndUtilization,
        'Demographics': RawDemographics,
        'Pharmacy Utilization': RawPharmacyUtilization,
        'Sales Channel': RawChannel,
        'Social Determinants of Health': RawSocialDeterminantsOfHealth,
        'Web Activity': RawWebActivity,
        'MEMBER_DATA': RawMemberData,
        'QUALITY_DATA': RawQualityData,
        'humana_mays_target_member_details': RawMemberDetails,
        'humana_mays_target_member_visit_claims': RawMemberClaims,
        'humana_mays_target_member_conditions': RawMemberCondition}
    if name == 'all':
        return dd
    else:
        return dd[name]

def main(args):
    """
    Build a database from CSVs.
    """
    # Load CSVs into DataFrames
    dataframes = {}

    def add_dataframes(csv_files, datas_type):
        print("Adding dataframes (type: '{}')".format(datas_type))
        for icsv, csv_path in enumerate(csv_files):
            name = csv_path.stem
            if name.endswith("_Holdout"): name = name[:-len("_Holdout")]
            print("[{}/{}] Reading csv for '{}' ({})".format(icsv+1, len(csv_files), name, csv_path.stem))
            df = pd.read_csv(csv_path)
            df['type'] = datas_type
            if name in dataframes:
                dataframes[name] = pd.concat([dataframes[name], df], axis=0)
            else:
                dataframes[name] = df
            print("  DF shape: {}".format(dataframes[name].shape))
            gc.collect()
        print()

    print("Loading training CSVs...")
    csv_root = Path(args.training_csvs_root_dir)
    csv_files = list(p for p in csv_root.iterdir() if p.suffix == ".csv")
    add_dataframes(csv_files, "Training")

    print("Loading holdout CSVs...")
    csv_root = Path(args.holdout_csvs_root_dir)
    csv_files = list(p for p in csv_root.iterdir() if p.suffix == ".csv")
    add_dataframes(csv_files, "Holdout")
    print("Loaded all CSVs into DataFrames")

    print("Identifying member ids...")
    ids_training = set(dataframes['humana_mays_target_members'][dataframes['humana_mays_target_members']['type'] == 'Training']['id'])
    ids_holdout = set(dataframes['humana_mays_target_members'][dataframes['humana_mays_target_members']['type'] == 'Holdout']['id'])
    assert len(ids_training.intersection(ids_holdout)) == 0
    print("Found {} member ids in training, and {} in holdout".format(len(ids_training), len(ids_holdout)))
    if args.test_run:
        print("This is a test run; sampling {} ids from training and holdout respectively.".format(args.test_run_n_ids))
        ids_training = set(random.sample(list(ids_training), args.test_run_n_ids))
        ids_holdout = set(random.sample(list(ids_holdout), args.test_run_n_ids))

    print("Verifying Training/Holdout status...")
    for name in tqdm(list(dataframes.keys())):
        df = dataframes[name]
        if not args.test_run:
            ids_training_cur = set(df[df['type'] == 'Training']['id'])
            ids_holdout_cur = set(df[df['type'] == 'Holdout']['id'])
            assert len(ids_training_cur) < len(ids_training) or ids_training_cur == ids_training
            assert len(ids_holdout_cur) < len(ids_holdout) or ids_holdout_cur == ids_holdout
        else:
            dataframes[name] = df[df['id'].isin(ids_training) | df['id'].isin(ids_holdout)]

    print("Initializing DB...")
    db_init()
    db_engine = get_db_engine()
    Session = sessionmaker(bind=db_engine)

    print("Inserting member info into DB...")
    with Session() as sess:
        for i, id_training in tqdm(list(enumerate(ids_training)), desc="Training"):
            e = Member()
            e.member_id = id_training
            e.data_type = EType.Training
            sess.add(e)
            if i > 0 and i % args.db_commit_interval == 0:
                sess.commit()
        sess.commit()
    
        for i, id_holdout in tqdm(list(enumerate(ids_holdout)), desc="Holdout"):
            e = Member()
            e.member_id = id_holdout
            e.data_type = EType.Holdout
            sess.add(e)
            if i > 0 and i % args.db_commit_interval == 0:
                sess.commit()
        sess.commit()

    # Get handlers & insert into DB
    params_all = {
        'QUALITY_DATA': {
            'base_event_date': {
                'date_fmt': "%d%b%Y"
            }
        },
        'humana_mays_target_member_visit_claims': {
            'serv_date_skey': {
                'date_fmt': "%Y%m%d"
            }
        },
    }
    handlers_all = {}
    for i, (name, df) in enumerate(dataframes.items()):
        print("Getting handlers [{}/{}] '{}'".format(i+1, len(dataframes), name))
        handlers = get_handlers(get_db_class(name), **params_all.get(name, {}))
        del handlers['id']
        print("  Handlers: {}".format(handlers))
        handlers_all[name] = handlers

    print("Start inserting into DB")
    for i, (name, df) in enumerate(dataframes.items()):
        db_class = get_db_class(name)
        print("Inserting into DB [{}/{}] '{}' (table: {}; {})".format(i+1, len(dataframes), name, db_class.__tablename__, df.shape))
        handlers = handlers_all[name]
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            db_item = db_class()
            for varname, h in handlers.items():
                try:
                    db_val = h.handle(row[varname if varname != 'member_id' else 'id'])
                except:
                    if isinstance(h, DateHandler):
                        db_val = None
                    else:
                        import traceback
                        traceback.print_exc()
                        breakpoint()
                        raise
                setattr(db_item, varname, db_val)
            sess.add(db_item)
            
            if i > 0 and i % args.db_commit_interval == 0:
                sess.commit()
        sess.commit()
    print("Done")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Build a database from CSVs")
    ap.add_argument('--training_csvs_root_dir', '-T', required=True, help="Directory containing CSVs for training")
    ap.add_argument('--holdout_csvs_root_dir', '-H', required=True, help="Directory containing holdout CSVs")
    ap.add_argument('--db_commit_interval', type=int, default=100)
    ap.add_argument('--test_run', action='store_true')
    ap.add_argument('--test_run_n_ids', type=int, default=50)
    args = ap.parse_args()
    main(args)
