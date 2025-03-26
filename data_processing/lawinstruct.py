"""LawInstruct"""

import json
import itertools
import random
import sys
from typing import NamedTuple, Optional
from absl import logging

import datasets

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
LawInstruct is an instruction tuning dataset of multilingual legal documents.
"""

_CITATION = """
"""

_URL = "https://hf-mirror.com/datasets/lawinstruct/lawinstruct"

_VERSION = datasets.Version("1.0.0", "")

# run python build_num_shards_dict.py to generate this dict
_NUM_SHARDS = {
    'BVADecisions-bva_decisions_label': 1,
    'BVADecisions-bva_decisions_qa': 1,
    'BrCAD5-brcad5_judgment': 1,
    'BrCAD5-brcad5_law_area': 1,
    'BrCAD5-brcad5_mc': 1,
    'BrCAD5-brcad5_topic': 1,
    'BrazilianBarExam-brazilian_bar_exam': 1,
    'CABarExamEssays-MainSubset': 1,
    'CAIL2019-cail_2019': 1,
    'CAIL2022-cail_2022_crime': 1,
    'CAIL2022-cail_2022_mc': 1,
    'CAIL2022-cail_2022_response': 1,
    # 'COLIEE-coliee_task3_generate_entailed_question': 1,  # Removed because of license issues
    # 'COLIEE-coliee_task3_passage_entailment': 1,  # Removed because of license issues
    # 'COLIEE-coliee_task4': 1,  # Removed because of license issues
    'CaseBriefs-case_briefs': 1,
    'ChangeMyView-change_my_view': 1,
    # 'CiviproQuestions-civipro_questions_generate_from_passage': 1,  # Removed because of license issues
    # 'CiviproQuestions-civipro_questions_generate_no_passage': 1,  # Removed because of license issues
    # 'CiviproQuestions-civipro_questions_no_explanation': 1,  # Removed because of license issues
    'ContractNLI-contract_nli': 1,
    'EOIRPrivacy-eoir_privacy': 1,
    'EdgarNER-MainSubset': 1,
    'Ell18GreekNER-MainSubset': 1,
    'Ell4GreekNER-MainSubset': 1,
    'EurLexSum-bulgarian': 1,
    'EurLexSum-croatian': 1,
    'EurLexSum-czech': 1,
    'EurLexSum-danish': 1,
    'EurLexSum-dutch': 1,
    'EurLexSum-english': 1,
    'EurLexSum-estonian': 1,
    'EurLexSum-finnish': 1,
    'EurLexSum-french': 1,
    'EurLexSum-german': 1,
    'EurLexSum-greek': 1,
    'EurLexSum-hungarian': 1,
    'EurLexSum-irish': 1,
    'EurLexSum-italian': 1,
    'EurLexSum-latvian': 1,
    'EurLexSum-lithuanian': 1,
    'EurLexSum-maltese': 1,
    'EurLexSum-polish': 1,
    'EurLexSum-portuguese': 1,
    'EurLexSum-romanian': 1,
    'EurLexSum-slovak': 1,
    'EurLexSum-slovenian': 1,
    'EurLexSum-spanish': 1,
    'EurLexSum-swedish': 1,
    'GermanLER-coarse': 1,
    'GermanLER-fine': 1,
    'GermanRentalAgreements-german_rental_agreements': 1,
    'ILDC-ildc': 1,
    'IndianNER-MainSubset': 1,
    'IndianTextSegmentation-indian_text_segmentation': 1,
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_acq': 1,
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_loss': 1,
    # 'JECQA-jec_qa': 1,  # Removed because of license issues
    'KoreanLegalQA-korean_legal_qa': 1,
    'LEXTREME-brazilian_court_decisions_judgment': 1,
    'LEXTREME-brazilian_court_decisions_unanimity': 1,
    'LEXTREME-covid19_emergency_event': 1,
    'LEXTREME-german_argument_mining': 1,
    'LEXTREME-greek_legal_code_chapter': 1,
    'LEXTREME-greek_legal_code_subject': 1,
    'LEXTREME-greek_legal_code_volume': 1,
    'LEXTREME-greek_legal_ner': 1,
    'LEXTREME-legalnero': 1,
    'LEXTREME-lener_br': 1,
    'LEXTREME-mapa_coarse': 1,
    'LEXTREME-mapa_fine': 1,
    'LEXTREME-multi_eurlex_level_1': 2,
    'LEXTREME-multi_eurlex_level_2': 1,
    'LEXTREME-multi_eurlex_level_3': 2,
    'LEXTREME-online_terms_of_service_clause_topics': 1,
    'LEXTREME-online_terms_of_service_unfairness_levels': 1,
    'LEXTREME-swiss_judgment_prediction': 1,
    'LawngNli-lawng_nli_entailment': 1,
    'LboxOpen-lbox_open_judgment': 1,
    'LboxOpen-lbox_open_statute': 1,
    'LegalCaseDocumentSummarization-legal_case_summarization_india': 1,
    'LegalCaseDocumentSummarization-legal_case_summarization_uk': 1,
    'LegalQA-legal_qa': 1,
    'LexGLUE-case_hold': 1,
    'LexGLUE-ecthr_a': 1,
    'LexGLUE-ecthr_b': 1,
    'LexGLUE-eurlex': 1,
    'LexGLUE-ledgar': 1,
    'LexGLUE-scotus': 1,
    'LexGLUE-unfair_tos': 1,
    'Littleton-littleton_events': 1,
    'Littleton-littleton_graph': 1,
    'MAUD-answer': 1,
    'MAUD-category': 1,
    'MAUD-question': 1,
    'MAUD-text_type': 1,
    # 'MBE-mbe_examples': 1,  # Removed because of license issues
    # 'MBE-mbe_subject': 1,  # Removed because of license issues
    # 'MBE-mbe_subject_generation': 1,  # Removed because of license issues
    'MCExamsLaw-mc_exams_law_explain': 1,
    'MCExamsLaw-mc_exams_law_no_explain': 1,
    'MiningLegalArguments-agent': 1,
    'MiningLegalArguments-argType': 1,
    'MultiLexSum-long_to_short': 1,
    'MultiLexSum-long_to_tiny': 1,
    'MultiLexSum-short_to_tiny': 1,
    'NaturalInstructionsLegal-billsum_summarization': 1,
    'NaturalInstructionsLegal-cail2018_answer_generation': 1,
    'NaturalInstructionsLegal-casehold_legal_answer_generation': 1,
    'NaturalInstructionsLegal-casehold_legal_incorrect_answer_generation': 1,
    'NaturalInstructionsLegal-cuad_answer_generation': 1,
    'NaturalInstructionsLegal-cuad_question_generation': 1,
    'NaturalInstructionsLegal-eurlex_classification': 1,
    'NaturalInstructionsLegal-eurlex_summarization': 1,
    'NaturalInstructionsLegal-online_privacy_policy_text_information_type_generation': 1,
    'NaturalInstructionsLegal-online_privacy_policy_text_purpose_answer_generation': 1,
    'NaturalInstructionsLegal-overruling_legal_classification': 1,
    'OLCMemos-olc_memos': 1,
    'PlainEnglishContractsSummarization-plain_english_contracts_summarization': 1,
    'PrivacyQA-privacy_qa': 1,
    'PrivacySummarization-privacy_summarization': 1,
    'RedditLegalQA-reddit_legal_qa': 1,
    'Sara-sara_entailment': 1,
    'Sara-sara_tax_liability': 1,
    'SaraProlog-sara_prolog_facts': 1,
    'SaraProlog-sara_prolog_statute': 1,
    'ShortAnswerFeedback-short_answer_feedback_error_class': 1,
    'ShortAnswerFeedback-short_answer_feedback_openqa': 1,
    'ShortAnswerFeedback-short_answer_feedback_rating': 1,
    'SpanishLaborLaw-spanish_labor_law': 1,
    'StackExchangeQuestionsLegal-stack_exchange_questions_legal': 1,
    'SwissCourtViewGeneration-swiss_judgment_court_view_generation_lower_court': 1,
    'SwissCourtViewGeneration-swiss_judgment_court_view_generation_same_court': 1,
    'SwissCriticalityPrediction-swiss_judgment_criticality': 1,
    'SwissJudgmentPrediction-swiss_judgment_multiple_choice': 1,
    'SwissJudgmentPredictionXL-swiss_judgment_dismiss_approve': 1,
    'SwissLawAreaPrediction-swiss_judgment_area_of_law_main_area': 1,
    'SwissLawAreaPrediction-swiss_judgment_area_of_law_sub_area': 1,
    'SwissLeadingDecisions-swiss_judgment_location': 1,
    'SwissLegislation-swiss_legislation_abbreviation': 1,
    'SwissLegislation-swiss_legislation_canton': 1,
    'SwissLegislation-swiss_legislation_short': 1,
    'SwissLegislation-swiss_legislation_title': 1,
    'TsccAlqac-tscc_alqac_question_answering': 1,
    'TurkishConstitutionalCourt-turkish_constitutional_multiple_choice': 1,
    'TurkishConstitutionalCourt-turkish_constitutional_violation_no_violation': 1,
    'USClassActions-us_class_actions_win_lose': 1,
    'ValidWills-valid_wills_entailment': 1
}

_ENGLISH = {
    'BVADecisions-bva_decisions_label',
    'BVADecisions-bva_decisions_qa',
    'CABarExamEssays-MainSubset',
    'CaseBriefs-case_briefs',
    'ChangeMyView-change_my_view',
    # 'CiviproQuestions-civipro_questions_generate_from_passage',  # Removed because of license issues
    # 'CiviproQuestions-civipro_questions_generate_no_passage',  # Removed because of license issues
    # 'CiviproQuestions-civipro_questions_no_explanation',  # Removed because of license issues
    'ContractNLI-contract_nli',
    'EdgarNER-MainSubset',
    'EOIRPrivacy-eoir_privacy',
    'EurLexSum-english',
    'ILDC-ildc',
    'IndianNER-MainSubset',
    'IndianTextSegmentation-indian_text_segmentation',
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_acq',
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_loss',
    'LawngNli-lawng_nli_entailment',
    'LegalCaseDocumentSummarization-legal_case_summarization_india',
    'LegalCaseDocumentSummarization-legal_case_summarization_uk',
    'LexGLUE-case_hold',
    'LexGLUE-ecthr_a',
    'LexGLUE-ecthr_b',
    'LexGLUE-eurlex',
    'LexGLUE-ledgar',
    'LexGLUE-scotus',
    'LexGLUE-unfair_tos',
    'Littleton-littleton_events',
    'Littleton-littleton_graph',
    'MAUD-answer',
    'MAUD-category',
    'MAUD-question',
    'MAUD-text_type',
    # 'MBE-mbe_examples',  # Removed because of license issues
    # 'MBE-mbe_subject',  # Removed because of license issues
    # 'MBE-mbe_subject_generation',  # Removed because of license issues
    'MCExamsLaw-mc_exams_law_explain',
    'MCExamsLaw-mc_exams_law_no_explain',
    'MiningLegalArguments-agent',
    'MiningLegalArguments-argType',
    'MultiLexSum-long_to_short',
    'MultiLexSum-long_to_tiny',
    'MultiLexSum-short_to_tiny',
    'NaturalInstructionsLegal-billsum_summarization',
    'NaturalInstructionsLegal-cail2018_answer_generation',
    'NaturalInstructionsLegal-casehold_legal_answer_generation',
    'NaturalInstructionsLegal-casehold_legal_incorrect_answer_generation',
    'NaturalInstructionsLegal-cuad_answer_generation',
    'NaturalInstructionsLegal-cuad_question_generation',
    'NaturalInstructionsLegal-eurlex_classification',
    'NaturalInstructionsLegal-eurlex_summarization',
    'NaturalInstructionsLegal-online_privacy_policy_text_information_type_generation',
    'NaturalInstructionsLegal-online_privacy_policy_text_purpose_answer_generation',
    'NaturalInstructionsLegal-overruling_legal_classification',
    'OLCMemos-olc_memos',
    'PlainEnglishContractsSummarization-plain_english_contracts_summarization',
    'PrivacyQA-privacy_qa',
    'PrivacySummarization-privacy_summarization',
    'RedditLegalQA-reddit_legal_qa',
    'Sara-sara_entailment',
    'Sara-sara_tax_liability',
    'SaraProlog-sara_prolog_facts',
    'SaraProlog-sara_prolog_statute',
    'StackExchangeQuestionsLegal-stack_exchange_questions_legal',
    'USClassActions-us_class_actions_win_lose',
    'ValidWills-valid_wills_entailment',
}

_COMMERCIAL = {
    'BVADecisions-bva_decisions_label',
    'BVADecisions-bva_decisions_qa',
    'EurLexSum-bulgarian',
    'EurLexSum-croatian',
    'EurLexSum-czech',
    'EurLexSum-danish',
    'EurLexSum-dutch',
    'EurLexSum-english',
    'EurLexSum-estonian',
    'EurLexSum-finnish',
    'EurLexSum-french',
    'EurLexSum-german',
    'EurLexSum-greek',
    'EurLexSum-hungarian',
    'EurLexSum-irish',
    'EurLexSum-italian',
    'EurLexSum-latvian',
    'EurLexSum-lithuanian',
    'EurLexSum-maltese',
    'EurLexSum-polish',
    'EurLexSum-portuguese',
    'EurLexSum-romanian',
    'EurLexSum-slovak',
    'EurLexSum-slovenian',
    'EurLexSum-spanish',
    'EurLexSum-swedish',
    'GermanLER-coarse',
    'GermanLER-fine',
    'IndianNER-MainSubset',
    'IndianTextSegmentation-indian_text_segmentation',
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_acq',
    'InternationalCitizenshipLawQuestions-international_citizenship_law_questions_mode_loss',
    'LawngNli-lawng_nli_entailment',
    'LegalCaseDocumentSummarization-legal_case_summarization_india',
    'LegalCaseDocumentSummarization-legal_case_summarization_uk',
    'LexGLUE-case_hold',
    'LexGLUE-ecthr_a',
    'LexGLUE-ecthr_b',
    'LexGLUE-eurlex',
    'LexGLUE-ledgar',
    'LexGLUE-scotus',
    'LexGLUE-unfair_tos',
    'LEXTREME-brazilian_court_decisions_judgment',
    'LEXTREME-brazilian_court_decisions_unanimity',
    'LEXTREME-covid19_emergency_event',
    'LEXTREME-german_argument_mining',
    'LEXTREME-greek_legal_code_chapter',
    'LEXTREME-greek_legal_code_subject',
    'LEXTREME-greek_legal_code_volume',
    'LEXTREME-greek_legal_ner',
    'LEXTREME-legalnero',
    'LEXTREME-lener_br',
    'LEXTREME-mapa_coarse',
    'LEXTREME-mapa_fine',
    'LEXTREME-multi_eurlex_level_1',
    'LEXTREME-multi_eurlex_level_2',
    'LEXTREME-multi_eurlex_level_3',
    'LEXTREME-online_terms_of_service_clause_topics',
    'LEXTREME-online_terms_of_service_unfairness_levels',
    'LEXTREME-swiss_judgment_prediction',
    'Littleton-littleton_events',
    'Littleton-littleton_graph',
    'MAUD-answer',
    'MAUD-category',
    'MAUD-question',
    'MAUD-text_type',
    'PrivacyQA-privacy_qa',
    'PrivacySummarization-privacy_summarization',
    'RedditLegalQA-reddit_legal_qa',
    'ShortAnswerFeedback-short_answer_feedback_error_class',
    'ShortAnswerFeedback-short_answer_feedback_openqa',
    'ShortAnswerFeedback-short_answer_feedback_rating',
    'SpanishLaborLaw-spanish_labor_law',
    'StackExchangeQuestionsLegal-stack_exchange_questions_legal',
    'SwissCourtViewGeneration-swiss_judgment_court_view_generation_lower_court',
    'SwissCourtViewGeneration-swiss_judgment_court_view_generation_same_court',
    'SwissCriticalityPrediction-swiss_judgment_criticality',
    'SwissJudgmentPrediction-swiss_judgment_multiple_choice',
    'SwissJudgmentPredictionXL-swiss_judgment_dismiss_approve',
    'SwissLawAreaPrediction-swiss_judgment_area_of_law_main_area',
    'SwissLawAreaPrediction-swiss_judgment_area_of_law_sub_area',
    'SwissLegislation-swiss_legislation_abbreviation',
    'SwissLegislation-swiss_legislation_canton',
    'SwissLegislation-swiss_legislation_short',
    'SwissLegislation-swiss_legislation_title',
    'TurkishConstitutionalCourt-turkish_constitutional_multiple_choice',
    'TurkishConstitutionalCourt-turkish_constitutional_violation_no_violation',
    'USClassActions-us_class_actions_win_lose',
}


class Instruction(NamedTuple):
    instruction: str
    lang: str


# Caching this function makes sense because it will typically be
# called with the same arguments repeatedly (i.e., for one dataset) before the
# next arguments arise.
# @functools.lru_cache(maxsize=16)
def _get_all_lang_instructions(
        group: str,
        all_instructions: dict[str, dict[str, list[str]]],
        size_per_lang: int = sys.maxsize,
        ilang: Optional[int] = None,
) -> list[tuple[str, str]]:
    """Combines individual-language instructions into one list."""
    # If self.mode == ilang, get the dict in all_instructions containing instructions in ilang
    if ilang is not None:
        all_instructions = {ilang: all_instructions[ilang]}

    result = [
        (instruction, lang)
        for lang, groups in all_instructions.items()
        for instruction in groups[group][:size_per_lang]
    ]
    # Make sure we didn't mess up the math.
    number_of_langs = len(all_instructions)
    assert len(result) <= size_per_lang * number_of_langs

    return result


class InstructionManager:
    """Class for managing instructions for different datasets."""

    def __init__(
            self,
            mode: str,
            instruction_bank_size: int,
            dl_manager: datasets.DownloadManager,
            random_state: Optional[int] = 42,
    ) -> None:
        """Creates an instruction bank that can be sampled from.

        Args:
            mode: whether English-only or multlingual instructions
            instruction_bank_size: number of instructions per
                (language x task_type) pair
            random_state: To ensure reproducibility
        """
        if mode == 'english':
            json_file = 'instruction_prompts/en.json'
        elif mode == 'multilingual' or mode == "ilang":
            json_file = 'instruction_prompts/multilingual.json'
        elif mode == 'dummy':
            json_file = 'instruction_prompts/dummy.json'
        else:
            raise ValueError(
                f'Mode {mode} should be "english", "multilingual", "ilang" or "dummy"')

        self._random = random.Random(random_state or 1337)
        self.mode = mode
        self._instruction_bank_size = instruction_bank_size
        self._instructions: dict[str, list[str]] = {}

        # Download instruction prompts json file locally
        local_json_file = dl_manager.download(json_file)
        # JSON file's structure is lang_code -> instruction_group -> text.
        with open(local_json_file) as f:
            self._instructions = json.load(f)

        self._confirm_well_formed(json_file, self._instructions)
        if not self._instructions or not all(self._instructions.values()):
            raise ValueError(
                f'Instruction bank {json_file} is empty or malformed.')

    def sample(self, task_type: str, ilang: Optional[str] = None) -> Instruction:
        """Sample an instruction (and its language) from the bank.

        Args:
            task_type: The name of the JSON field with relevant instructions.
            ilang: Optional argument for the input language if self.mode == ilang

        Returns:
            A 2-tuple: instruction text and its two-letter language code.
        """
        universe = _get_all_lang_instructions(
            task_type,
            self._instructions,
            self._instruction_bank_size,
            ilang,
        )
        instruction, lang = self._random.choice(universe)
        return Instruction(instruction=instruction, lang=lang)

    def _confirm_well_formed(
            self,
            json_file: str,
            instructions: dict[str, dict[str, list[str]]],
    ) -> None:
        # Check that there are any languages.
        if not instructions:
            raise ValueError(f'Instruction bank {json_file} is empty.')
        # Check that all languages have at least one instruction group.
        if not all(instructions.values()):
            empty = {
                lang
                for lang, groups in instructions.items()
                if not groups
            }
            raise ValueError(
                f'No instructions for language(s) {empty} in {json_file}.')
        # Check that all instruction groups are non-empty.
        for lang, groups in instructions.items():
            if not all(groups.values()):
                empty = {
                    group
                    for group, options in groups.items()
                    if not options
                }
                raise ValueError(
                    f'No instructions for group(s) {empty}'
                    f' in language {lang} in {json_file}')
        # Check that all instructions are non-empty and warn if they might be bad f-strings.
        for lang, groups in instructions.items():
            for group, options in groups.items():
                if not all(options):
                    raise ValueError(
                        f'Found empty instruction for group {group} in language'
                        f' {lang} in {json_file}')
                if any('{' in option for option in options):
                    for option in options:
                        if '{' in option:
                            logging.warning('Open-brace present in instruction. '
                                            'Was an f-string moved into the JSON incorrectly? See: %s', option)


class LawInstructConfig(datasets.BuilderConfig):
    """BuilderConfig for LawInstruct."""

    def __init__(self, name: str, **kwargs):
        """BuilderConfig for LawInstruct.
        Args:
            name: the name of the sub dataset
          **kwargs: keyword arguments forwarded to super.
        """
        super(LawInstructConfig, self).__init__(**kwargs)
        self.name = name

        # Get the instruction sampling option
        instruction_sampling_option = self.name.split('-')[-1]
        instruction_bank_size, lang = instruction_sampling_option.split('_')
        instruction_bank_size = int(instruction_bank_size)
        if lang == "english":
            mode = "english"
        elif lang == "multi":
            mode = "multilingual"
        elif lang == "ilang":
            mode = "ilang"
        else:
            raise ValueError(
                f'Instruction sampling language option {lang} should be "english", "ilang", or "multi"')
        self.mode = mode
        self.instruction_bank_size = instruction_bank_size


class LawInstruct(datasets.GeneratorBasedBuilder):
    """LawInstruct: An instruction tuning dataset of multilingual legal documents."""
    BUILDER_CONFIG_CLASS = LawInstructConfig

    dataset_options = ["all_multi", "commercial_multi", "all_english", "commercial_english"] + list(_NUM_SHARDS.keys())
    instruction_sampling_options = ["1_english", "10_english", "10_ilang", "10_multi"]
    config_tuples = itertools.product(dataset_options, instruction_sampling_options)
    names = ['-'.join(p) for p in config_tuples]
    BUILDER_CONFIGS = [LawInstructConfig(name, version=_VERSION, description=f"{name} subset of LawInstruct") for name
                       in names]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "dataset_name": datasets.Value("string"),
                    "subset_name": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "instruction_language": datasets.Value("string"),
                    "prompt_language": datasets.Value("string"),
                    "answer_language": datasets.Value("string"),
                    "jurisdiction": datasets.Value("string"),
                    "task_type": datasets.Value("string"),
                    "downloaded_timestamp": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_infos = []
        if self.config.name.startswith("all_multi"):
            names = set(_NUM_SHARDS.keys())
        elif self.config.name.startswith("all_english"):
            names = _ENGLISH
        elif self.config.name.startswith("commercial_multi"):
            names = _COMMERCIAL
        elif self.config.name.startswith("commercial_english"):
            names = _COMMERCIAL.intersection(_ENGLISH)
        else:
            # Split off the instruction sampling option from self.config.name
            names = ["-".join(self.config.name.split("-")[0:-1])]
        names = list(names)

        for name in names:
            for shard in range(_NUM_SHARDS[name]):
                data_infos.append({"filepath": dl_manager.download(f"data/{name}-train-{shard}.jsonl.xz")})

        # Initialize the instruction sampling manager
        instructions = InstructionManager(
            mode=self.config.mode,
            instruction_bank_size=self.config.instruction_bank_size,
            dl_manager=dl_manager,
        )

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={"data_infos": data_infos, "instructions": instructions})]

    def _generate_examples(self, data_infos, instructions):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        # Get the subset name keys in the instructions dictionary
        subsets = list(instructions._instructions.values())[0].keys()

        id_ = 0
        for data_info in data_infos:
            logger.info("Generating examples from = %s", data_info["filepath"])
            try:
                with xz.open(open(data_info["filepath"], "rb"), "rt", encoding="utf-8") as f:
                    for line in f:
                        if line:
                            example = json.loads(line)
                            if example is not None and isinstance(example, dict):
                                instruction = example.get("instruction", "")
                                subset_name = example.get("subset_name", "")
                                # Don't sample if either the instruction in a blank instruction (e.g., CABarExamEssays)
                                # or the instruction is tied to task in a specific way that should not be sampled (e.g., NER)
                                if instruction != "" and subset_name in subsets:
                                    if self.config.mode == "ilang":
                                        # If there's no input (prompt) language, default to "en""
                                        ilang = example.get("prompt_language", "en")
                                        instruction, instruction_language = instructions.sample(subset_name, ilang)
                                    else:
                                        instruction, instruction_language = instructions.sample(subset_name)
                                else:
                                    instruction_language = example.get("instruction_language", "")

                                yield id_, {
                                    "dataset_name": example.get("dataset_name", ""),
                                    "subset_name": subset_name,
                                    "source": example.get("source", ""),
                                    "instruction_language": instruction_language,
                                    "prompt_language": example.get("prompt_language", ""),
                                    "answer_language": example.get("answer_language", ""),
                                    "jurisdiction": example.get("jurisdiction", ""),
                                    "task_type": example.get("task_type", ""),
                                    "downloaded_timestamp": example.get("downloaded_timestamp", ""),
                                    "instruction": instruction,
                                    "prompt": example.get("prompt", ""),
                                    "answer": example.get("answer", ""),
                                }
                                id_ += 1
            except Exception:
                logger.exception("Error while processing file %s", data_info["filepath"])
