
# print
from services.printing.print_phase import print_sub_phase_start, print_sub_phase_end

def run_phase(phase_settings):

    print_sub_phase_start(phase_settings.name)
    phase_settings.function()
    print_sub_phase_end(phase_settings.name)
