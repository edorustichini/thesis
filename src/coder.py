import os
from config import setup_parser
from jpegai_compress_directory import RecoEncoder, create_custom_parser, def_encoder_parser_decorator, get_downloader

class CoderManager:
    """
    Class for coder setup and managment
    """
    def __init__(self, args):
        self.args = args or setup_parser()
        self.coder = None
        self.setup_device()
        self.setup_coder()
    
    def setup_device(self):
        if self.args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            print(f"Using GPU {self.args.gpu}")
            self.args.target_device = 'gpu'
        else:
            self.args.target_device = 'cpu'
            print("Using CPU")
    
    def setup_coder(self):
        encoder_parser = create_custom_parser(self.args)
        
        # --- Setup the coder --- #
        encoder_parser = create_custom_parser(self.args)
        
        # --- Setup the encoder --- #
        self.coder = RecoEncoder(encoder_parser, def_encoder_parser_decorator(encoder_parser))
        
        # --- Setup the coding engine --- #
        self.coder.print_coder_info()
        
        kwargs, params, _ = self.coder.init_common_codec(
            build_model=True,
            ce=None,
            cmd_args=None,
            overload_ce=True,
            cmd_args_add=False
        )
        
        # Load the models
        self.coder.load_models(
            get_downloader(
                kwargs.get('models_dir_name', 'models'),
                critical_for_file_absence=not kwargs.get('skip_loading_error', False)
            )
        )
        self.coder.set_target_bpp_idx(kwargs['bpp_idx'])
        
        print("Coder setup completed successfully")


    
    