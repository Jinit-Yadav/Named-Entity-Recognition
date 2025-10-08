import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True

            # Use the config DATA_DIR
            data_dir = self.config.DATA_DIR
            
            # Check if the data directory exists
            if not os.path.exists(data_dir):
                logger.error(f"Data directory not found: {data_dir}")
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                return validation_status

            all_files = os.listdir(data_dir)
            logger.info(f"Files found in {data_dir}: {all_files}")

            # Check for each required file
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    logger.error(f"Missing required file: {required_file}")
                    break
                else:
                    logger.info(f"Found required file: {required_file}")

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data validation completed: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: False")
            raise e