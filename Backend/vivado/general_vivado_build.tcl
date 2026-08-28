# ============================================================
# Generic ZynqMP + Multi-HLS Vivado Block Design Generator
# Supports:
#   - multiple HLS IPs
#   - multiple AXI-MM DDR ports
#   - AXI-Stream links
#   - interrupts
#   - multiple clock/reset mappings
# ============================================================

# ============================================================
# Default configuration
# ============================================================

dict set CONFIG PROJECT_NAME "lightcnn_subfunction_c_viv"
dict set CONFIG PROJECT_DIR  "[pwd]/build/vivado"

dict set CONFIG PART       "xczu7ev-ffvc1156-2-e"
dict set CONFIG BOARD_PART "xilinx.com:zcu104:part0:1.1"

dict set CONFIG BD_NAME "design_1"

dict set CONFIG IP_REPO "[pwd]/build/ip_repo"

dict set CONFIG PS_NAME "zynq_ultra_ps_e_0"
dict set CONFIG RESET_NAME "proc_sys_reset_0"
dict set CONFIG CTRL_IC_NAME "axi_control_ic"

dict set CONFIG PS_CTRL_PORT "M_AXI_HPM0_FPD"
dict set CONFIG PS_CTRL_ACLK "maxihpm0_fpd_aclk"

dict set CONFIG DEFAULT_PL_CLK "pl_clk0"
dict set CONFIG DEFAULT_PL_RESET "pl_resetn0"

dict set CONFIG JOBS 96
dict set CONFIG RUN_SYNTH 1
dict set CONFIG RUN_IMPL  1
dict set CONFIG GENERATE_REPORTS 1
dict set CONFIG REPORT_DIR ""

# ------------------------------------------------------------
# HLS_SPECS format:
#
# name|vlnv|ctrl_if|data_ports|clk|reset|irq
#
# Multiple HLS IPs separated by semicolon.
#
# Example:
# HLS_SPECS='kernel0|xilinx.com:hls:k0:1.0|s_axi_control|m_axi_a,m_axi_b|ap_clk|ap_rst_n|interrupt;kernel1|xilinx.com:hls:k1:1.0|s_axi_CTRL|m_axi_in,m_axi_out|aclk|aresetn|'
# ------------------------------------------------------------
dict set CONFIG HLS_SPECS "lightcnn_sub_c_0|xilinx.com:hls:lightcnn_sub_c:1.0|s_axi_CTRL|m_axi_a,m_axi_b|ap_clk|ap_rst_n|"

# ------------------------------------------------------------
# AXI-MM DDR map format:
#
# hls_ip.port=PS_DDR_PORT
#
# Example:
# AXI_MM_MAP='kernel0.m_axi_a=S_AXI_HP0_FPD,kernel0.m_axi_b=S_AXI_HP1_FPD,kernel1.m_axi_out=S_AXI_HPC0_FPD'
#
# If empty, HLS IPs are distributed automatically across PS_DDR_PORTS,
# with at most 8 HLS IPs (16 AXI masters) per interconnect.
# ------------------------------------------------------------
dict set CONFIG AXI_MM_MAP ""
dict set CONFIG DEFAULT_PS_DDR_PORT "S_AXI_HP0_FPD"

# Ordered pool used when AXI_MM_MAP is empty.  Each data interconnect is
# attached to the next free PS port and carries at most 8 two-master HLS IPs.
dict set CONFIG PS_DDR_PORTS "S_AXI_HP0_FPD,S_AXI_HP1_FPD,S_AXI_HP2_FPD,S_AXI_HP3_FPD,S_AXI_HPC0_FPD,S_AXI_HPC1_FPD"

# ------------------------------------------------------------
# PS DDR clock map
# ------------------------------------------------------------
dict set CONFIG PS_DDR_ACLK_MAP "S_AXI_HP0_FPD=saxihp0_fpd_aclk,S_AXI_HP1_FPD=saxihp1_fpd_aclk,S_AXI_HP2_FPD=saxihp2_fpd_aclk,S_AXI_HP3_FPD=saxihp3_fpd_aclk,S_AXI_HPC0_FPD=saxihpc0_fpd_aclk,S_AXI_HPC1_FPD=saxihpc1_fpd_aclk"

# Vivado PS properties which enable the corresponding DDR-facing ports.
dict set CONFIG PS_DDR_ENABLE_MAP "S_AXI_HP0_FPD=CONFIG.PSU__USE__S_AXI_GP2,S_AXI_HP1_FPD=CONFIG.PSU__USE__S_AXI_GP3,S_AXI_HP2_FPD=CONFIG.PSU__USE__S_AXI_GP4,S_AXI_HP3_FPD=CONFIG.PSU__USE__S_AXI_GP5,S_AXI_HPC0_FPD=CONFIG.PSU__USE__S_AXI_GP0,S_AXI_HPC1_FPD=CONFIG.PSU__USE__S_AXI_GP1"

# ------------------------------------------------------------
# PS enable properties
#
# You must enable the PS ports you use.
# These defaults match the original LightCNN design.
# ------------------------------------------------------------
dict set CONFIG PS_ENABLE_PROPS "CONFIG.PSU__USE__M_AXI_GP0=1,CONFIG.PSU__USE__M_AXI_GP1=0,CONFIG.PSU__USE__S_AXI_GP2=1"

# ------------------------------------------------------------
# AXI-Stream links format:
#
# src_ip.src_axis=dst_ip.dst_axis
#
# Example:
# AXIS_LINKS='kernel0.out_r=kernel1.in_r,kernel1.out_r=kernel2.in_r'
# ------------------------------------------------------------
dict set CONFIG AXIS_LINKS ""

# ------------------------------------------------------------
# Clock map format:
#
# hls_ip=pl_clkN
#
# Example:
# CLOCK_MAP='kernel0=pl_clk0,kernel1=pl_clk1'
# ------------------------------------------------------------
dict set CONFIG CLOCK_MAP ""

# ------------------------------------------------------------
# Reset map format:
#
# hls_ip=pl_resetnN
#
# Example:
# RESET_MAP='kernel0=pl_resetn0,kernel1=pl_resetn1'
# ------------------------------------------------------------
dict set CONFIG RESET_MAP ""

# ------------------------------------------------------------
# Interrupt configuration
# If IRQ ports are specified in HLS_SPECS, they are connected
# through xlconcat to pl_ps_irq0.
# ------------------------------------------------------------
dict set CONFIG IRQ_CONCAT_NAME "xlconcat_irq"
dict set CONFIG PS_IRQ_PORT "pl_ps_irq0"

# Optional external config file
dict set CONFIG CONFIG_FILE ""

# ============================================================
# Override CONFIG from command line
# ============================================================

foreach arg $argv {
    if {[regexp {([^=]+)=(.*)} $arg -> key value]} {
        dict set CONFIG $key $value
    }
}

if {[dict get $CONFIG CONFIG_FILE] ne ""} {
    source [dict get $CONFIG CONFIG_FILE]
}

if {[dict get $CONFIG REPORT_DIR] eq ""} {
    dict set CONFIG REPORT_DIR [file join [dict get $CONFIG PROJECT_DIR] reports]
}

# ============================================================
# Helper procedures
# ============================================================

proc C {key} {
    global CONFIG
    return [dict get $CONFIG $key]
}

proc bd_pin {cell pin} {
    return [get_bd_pins "${cell}/${pin}"]
}

proc bd_intf {cell pin} {
    return [get_bd_intf_pins "${cell}/${pin}"]
}

proc maybe_connect_net {src dst} {
    if {[llength $src] && [llength $dst]} {
        connect_bd_net $src $dst
    }
}

proc parse_kv_list {s} {
    set d [dict create]

    if {$s eq ""} {
        return $d
    }

    foreach item [split $s ","] {
        if {[regexp {([^=]+)=(.*)} $item -> k v]} {
            dict set d $k $v
        }
    }

    return $d
}

proc parse_hls_specs {s} {
    set result {}

    foreach spec [split $s ";"] {
        if {$spec eq ""} {
            continue
        }

        set f [split $spec "|"]

        if {[llength $f] < 7} {
            error "Bad HLS_SPECS entry: $spec"
        }

        set d [dict create]
        dict set d name       [lindex $f 0]
        dict set d vlnv       [lindex $f 1]
        dict set d ctrl_if    [lindex $f 2]
        dict set d data_ports [lindex $f 3]
        dict set d clk        [lindex $f 4]
        dict set d rst        [lindex $f 5]
        dict set d irq        [lindex $f 6]

        lappend result $d
    }

    return $result
}

proc sanitize_name {s} {
    regsub -all {[^a-zA-Z0-9_]} $s "_" out
    return $out
}

proc first_or_default {dict_obj key default} {
    if {[dict exists $dict_obj $key]} {
        return [dict get $dict_obj $key]
    }
    return $default
}

# ============================================================
# Derived data
# ============================================================

set HLS_LIST        [parse_hls_specs [C HLS_SPECS]]
set AXI_MM_MAP_D    [parse_kv_list [C AXI_MM_MAP]]
set PS_DDR_ACLK_D   [parse_kv_list [C PS_DDR_ACLK_MAP]]
set PS_DDR_ENABLE_D [parse_kv_list [C PS_DDR_ENABLE_MAP]]
set CLOCK_MAP_D     [parse_kv_list [C CLOCK_MAP]]
set RESET_MAP_D     [parse_kv_list [C RESET_MAP]]
set PS_ENABLE_D     [parse_kv_list [C PS_ENABLE_PROPS]]

dict set CONFIG TOP_NAME "[C BD_NAME]_wrapper"

set num_hls [llength $HLS_LIST]

if {$num_hls == 0} {
    error "HLS_SPECS must contain at least one HLS IP"
}

# Build the data-interconnect plan before creating the PS so that every PS
# slave port selected by the plan can be enabled automatically.
set data_ic_groups {}

if {[C AXI_MM_MAP] eq ""} {
    set ps_ddr_ports [split [C PS_DDR_PORTS] ","]
    set num_data_ics [expr {($num_hls + 7) / 8}]

    if {[llength $ps_ddr_ports] != [llength [lsort -unique $ps_ddr_ports]]} {
        error "PS_DDR_PORTS must contain unique PS slave ports"
    }

    if {$num_data_ics > [llength $ps_ddr_ports]} {
        error "$num_hls HLS IPs require $num_data_ics data interconnects, but PS_DDR_PORTS contains only [llength $ps_ddr_ports] free PS ports"
    }

    for {set group_index 0} {$group_index < $num_data_ics} {incr group_index} {
        set refs {}
        set first_hls [expr {$group_index * 8}]
        set last_hls [expr {min($first_hls + 7, $num_hls - 1)}]

        for {set hls_index $first_hls} {$hls_index <= $last_hls} {incr hls_index} {
            set hls [lindex $HLS_LIST $hls_index]
            set name [dict get $hls name]
            set data_ports [split [dict get $hls data_ports] ","]

            if {[llength $data_ports] != 2} {
                error "HLS IP $name must have exactly two AXI master data interfaces"
            }

            foreach port $data_ports {
                lappend refs "$name.$port"
            }
        }

        lappend data_ic_groups [dict create \
            ddr_port [lindex $ps_ddr_ports $group_index] \
            refs $refs]
    }
} else {
    # Preserve the original explicit mapping behavior: one interconnect per
    # mapped PS port.  Explicit groups must still fit the hardware's 16 SIs.
    set ddr_groups [dict create]

    foreach hls $HLS_LIST {
        set name [dict get $hls name]
        set ports_str [dict get $hls data_ports]

        if {$ports_str eq ""} {
            continue
        }

        foreach port [split $ports_str ","] {
            set ref "$name.$port"
            set ddr_port [first_or_default $AXI_MM_MAP_D $ref [C DEFAULT_PS_DDR_PORT]]
            dict lappend ddr_groups $ddr_port $ref
        }
    }

    foreach ddr_port [dict keys $ddr_groups] {
        set refs [dict get $ddr_groups $ddr_port]

        if {[llength $refs] > 16} {
            error "Explicit AXI_MM_MAP assigns [llength $refs] masters to $ddr_port; an AXI Interconnect supports at most 16"
        }

        lappend data_ic_groups [dict create ddr_port $ddr_port refs $refs]
    }
}

foreach group $data_ic_groups {
    set ddr_port [dict get $group ddr_port]

    if {![dict exists $PS_DDR_ACLK_D $ddr_port]} {
        error "No PS_DDR_ACLK_MAP entry for data port $ddr_port"
    }
    if {![dict exists $PS_DDR_ENABLE_D $ddr_port]} {
        error "No PS_DDR_ENABLE_MAP entry for data port $ddr_port"
    }

    dict set PS_ENABLE_D [dict get $PS_DDR_ENABLE_D $ddr_port] 1
}

# ============================================================
# Print configuration
# ============================================================

puts "============================================================"
puts "Active Vivado configuration"
puts "============================================================"

foreach key [lsort [dict keys $CONFIG]] {
    puts "$key = [dict get $CONFIG $key]"
}

puts "============================================================"

# ============================================================
# Create project
# ============================================================

create_project [C PROJECT_NAME] [C PROJECT_DIR] -part [C PART] -force
set_property board_part [C BOARD_PART] [current_project]

set_property ip_repo_paths [C IP_REPO] [current_project]
update_ip_catalog -rebuild

create_bd_design [C BD_NAME]

# ============================================================
# Create PS
# ============================================================

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 [C PS_NAME]

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells [C PS_NAME]]

set ps_prop_list {}

foreach k [dict keys $PS_ENABLE_D] {
    lappend ps_prop_list $k [dict get $PS_ENABLE_D $k]
}

if {[llength $ps_prop_list] > 0} {
    set_property -dict $ps_prop_list [get_bd_cells [C PS_NAME]]
}

# ============================================================
# Create reset IP
# ============================================================

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 [C RESET_NAME]

set default_clk [bd_pin [C PS_NAME] [C DEFAULT_PL_CLK]]
set default_rst [bd_pin [C PS_NAME] [C DEFAULT_PL_RESET]]

connect_bd_net $default_clk [bd_pin [C RESET_NAME] slowest_sync_clk]
connect_bd_net $default_rst [bd_pin [C RESET_NAME] ext_reset_in]

set rstn [bd_pin [C RESET_NAME] interconnect_aresetn]
set periph_rstn [bd_pin [C RESET_NAME] peripheral_aresetn]

# ============================================================
# Create HLS IPs
# ============================================================

set hls_names {}
set irq_sources {}

foreach hls $HLS_LIST {
    set name [dict get $hls name]
    set vlnv [dict get $hls vlnv]

    puts "Creating HLS IP: $name $vlnv"

    create_bd_cell -type ip -vlnv $vlnv $name
    lappend hls_names $name
}

# ============================================================
# AXI-Lite control interconnect(s)
# One PS master -> up to 16 HLS control slaves per leaf interconnect
# ============================================================

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 [C CTRL_IC_NAME]

set num_ctrl_leaves [expr {($num_hls + 15) / 16}]
set use_ctrl_leaves [expr {$num_ctrl_leaves > 1}]
set root_num_mi [expr {$use_ctrl_leaves ? $num_ctrl_leaves : $num_hls}]

set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI $root_num_mi] \
    [get_bd_cells [C CTRL_IC_NAME]]

connect_bd_intf_net \
    [bd_intf [C PS_NAME] [C PS_CTRL_PORT]] \
    [bd_intf [C CTRL_IC_NAME] S00_AXI]

connect_bd_net $default_clk [bd_pin [C PS_NAME] [C PS_CTRL_ACLK]]
connect_bd_net $default_clk [bd_pin [C CTRL_IC_NAME] ACLK]
connect_bd_net $default_clk [bd_pin [C CTRL_IC_NAME] S00_ACLK]

connect_bd_net $rstn [bd_pin [C CTRL_IC_NAME] ARESETN]
connect_bd_net $rstn [bd_pin [C CTRL_IC_NAME] S00_ARESETN]

set ctrl_index 0

if {$use_ctrl_leaves} {
    for {set leaf_index 0} {$leaf_index < $num_ctrl_leaves} {incr leaf_index} {
        set leaf_name "[C CTRL_IC_NAME]_leaf_${leaf_index}"
        set leaf_num_mi [expr {min(16, $num_hls - ($leaf_index * 16))}]
        set root_mi_if [format "M%02d_AXI" $leaf_index]
        set root_mi_clk [format "M%02d_ACLK" $leaf_index]
        set root_mi_rstn [format "M%02d_ARESETN" $leaf_index]

        create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 $leaf_name
        set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI $leaf_num_mi] \
            [get_bd_cells $leaf_name]

        connect_bd_intf_net [bd_intf [C CTRL_IC_NAME] $root_mi_if] [bd_intf $leaf_name S00_AXI]
        connect_bd_net $default_clk [bd_pin [C CTRL_IC_NAME] $root_mi_clk] [bd_pin $leaf_name ACLK] [bd_pin $leaf_name S00_ACLK]
        connect_bd_net $rstn [bd_pin [C CTRL_IC_NAME] $root_mi_rstn] [bd_pin $leaf_name ARESETN] [bd_pin $leaf_name S00_ARESETN]
    }
}

foreach hls $HLS_LIST {
    set name    [dict get $hls name]
    set ctrl_if [dict get $hls ctrl_if]

    if {$use_ctrl_leaves} {
        set ctrl_ic "[C CTRL_IC_NAME]_leaf_[expr {$ctrl_index / 16}]"
        set local_ctrl_index [expr {$ctrl_index % 16}]
    } else {
        set ctrl_ic [C CTRL_IC_NAME]
        set local_ctrl_index $ctrl_index
    }

    set mi_if   [format "M%02d_AXI" $local_ctrl_index]
    set mi_clk  [format "M%02d_ACLK" $local_ctrl_index]
    set mi_rstn [format "M%02d_ARESETN" $local_ctrl_index]

    puts "Connecting control: $ctrl_ic/$mi_if -> $name/$ctrl_if"

    connect_bd_intf_net \
        [bd_intf $ctrl_ic $mi_if] \
        [bd_intf $name $ctrl_if]

    connect_bd_net $default_clk [bd_pin $ctrl_ic $mi_clk]
    connect_bd_net $rstn       [bd_pin $ctrl_ic $mi_rstn]

    incr ctrl_index
}

# ============================================================
# Build AXI-MM DDR interconnects from the precomputed groups
# ============================================================

set data_ic_index 0

foreach group $data_ic_groups {
    set ddr_port [dict get $group ddr_port]
    set refs [dict get $group refs]
    set num_si [llength $refs]

    set ic_name "axi_data_ic_${data_ic_index}_[sanitize_name $ddr_port]"

    puts "Creating data interconnect $ic_name for $ddr_port with $num_si slave inputs"

    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 $ic_name

    set_property -dict [list \
        CONFIG.NUM_SI $num_si \
        CONFIG.NUM_MI 1 \
    ] [get_bd_cells $ic_name]

    connect_bd_intf_net \
        [bd_intf $ic_name M00_AXI] \
        [bd_intf [C PS_NAME] $ddr_port]

    if {[dict exists $PS_DDR_ACLK_D $ddr_port]} {
        set ps_ddr_aclk [dict get $PS_DDR_ACLK_D $ddr_port]
        connect_bd_net $default_clk [bd_pin [C PS_NAME] $ps_ddr_aclk]
    }

    connect_bd_net $default_clk [bd_pin $ic_name ACLK]
    connect_bd_net $default_clk [bd_pin $ic_name M00_ACLK]

    connect_bd_net $rstn [bd_pin $ic_name ARESETN]
    connect_bd_net $rstn [bd_pin $ic_name M00_ARESETN]

    set si_index 0

    foreach ref $refs {
        regexp {([^\.]+)\.(.+)} $ref -> hls_name port_name

        set si_if   [format "S%02d_AXI" $si_index]
        set si_clk  [format "S%02d_ACLK" $si_index]
        set si_rstn [format "S%02d_ARESETN" $si_index]

        puts "Connecting data: $hls_name/$port_name -> $ic_name/$si_if -> $ddr_port"

        connect_bd_intf_net \
            [bd_intf $hls_name $port_name] \
            [bd_intf $ic_name $si_if]

        connect_bd_net $default_clk [bd_pin $ic_name $si_clk]
        connect_bd_net $rstn       [bd_pin $ic_name $si_rstn]

        incr si_index
    }

    incr data_ic_index
}

# ============================================================
# Clock/reset connections for HLS IPs
# ============================================================

foreach hls $HLS_LIST {
    set name [dict get $hls name]
    set clk_pin [dict get $hls clk]
    set rst_pin [dict get $hls rst]

    set pl_clk   [first_or_default $CLOCK_MAP_D $name [C DEFAULT_PL_CLK]]
    set pl_reset [first_or_default $RESET_MAP_D $name [C DEFAULT_PL_RESET]]

    puts "Connecting clock/reset for $name: $pl_clk -> $clk_pin, reset -> $rst_pin"

    connect_bd_net \
        [bd_pin [C PS_NAME] $pl_clk] \
        [bd_pin $name $clk_pin]

    connect_bd_net \
        $periph_rstn \
        [bd_pin $name $rst_pin]

    set irq_pin [dict get $hls irq]

    if {$irq_pin ne ""} {
        lappend irq_sources "$name.$irq_pin"
    }
}

# ============================================================
# AXI-Stream links
# ============================================================

if {[C AXIS_LINKS] ne ""} {
    foreach link [split [C AXIS_LINKS] ","] {
        if {![regexp {([^=]+)=(.+)} $link -> src dst]} {
            error "Bad AXIS_LINKS entry: $link"
        }

        regexp {([^\.]+)\.(.+)} $src -> src_cell src_port
        regexp {([^\.]+)\.(.+)} $dst -> dst_cell dst_port

        puts "Connecting AXI-Stream: $src_cell/$src_port -> $dst_cell/$dst_port"

        connect_bd_intf_net \
            [bd_intf $src_cell $src_port] \
            [bd_intf $dst_cell $dst_port]
    }
}

# ============================================================
# Interrupts
# ============================================================

set num_irqs [llength $irq_sources]

if {$num_irqs > 0} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 [C IRQ_CONCAT_NAME]

    set_property -dict [list CONFIG.NUM_PORTS $num_irqs] \
        [get_bd_cells [C IRQ_CONCAT_NAME]]

    set irq_index 0

    foreach irq_ref $irq_sources {
        regexp {([^\.]+)\.(.+)} $irq_ref -> cell pin

        set concat_pin [format "In%d" $irq_index]

        puts "Connecting IRQ: $cell/$pin -> [C IRQ_CONCAT_NAME]/$concat_pin"

        connect_bd_net \
            [bd_pin $cell $pin] \
            [bd_pin [C IRQ_CONCAT_NAME] $concat_pin]

        incr irq_index
    }

    connect_bd_net \
        [bd_pin [C IRQ_CONCAT_NAME] dout] \
        [bd_pin [C PS_NAME] [C PS_IRQ_PORT]]
}

# ============================================================
# Address assignment and validation
# ============================================================

assign_bd_address
validate_bd_design
save_bd_design

# ============================================================
# Generate output products and wrapper
# ============================================================

generate_target all [get_files [C BD_NAME].bd]

make_wrapper -files [get_files [C BD_NAME].bd] -top

set WRAPPER_PATH "[C PROJECT_DIR]/[C PROJECT_NAME].gen/sources_1/bd/[C BD_NAME]/hdl/[C TOP_NAME].v"

if {![file exists $WRAPPER_PATH]} {
    set candidates [glob -nocomplain "[C PROJECT_DIR]/[C PROJECT_NAME].gen/sources_1/bd/[C BD_NAME]/hdl/*wrapper.v"]

    if {[llength $candidates] == 0} {
        error "Could not find generated HDL wrapper."
    }

    set WRAPPER_PATH [lindex $candidates 0]
}

add_files -norecurse $WRAPPER_PATH

set_property top [C TOP_NAME] [current_fileset]
update_compile_order -fileset sources_1

# ============================================================
# Build
# ============================================================

if {[C RUN_SYNTH]} {
    launch_runs synth_1 -jobs [C JOBS]
    wait_on_run synth_1
}

if {[C RUN_IMPL]} {
    launch_runs impl_1 -to_step write_bitstream -jobs [C JOBS]
    wait_on_run impl_1

    if {[C GENERATE_REPORTS]} {
        set impl_status [get_property STATUS [get_runs impl_1]]

        if {![string match "*Complete*" $impl_status]} {
            error "Implementation did not complete; cannot generate reports (status: $impl_status)"
        }

        file mkdir [C REPORT_DIR]
        open_run impl_1

        set utilization_report [file join [C REPORT_DIR] module_utilization.csv]
        set power_report [file join [C REPORT_DIR] power_report.csv]
        set timing_report [file join [C REPORT_DIR] timing_paths.csv]

        report_utilization \
            -hierarchical \
            -hierarchical_depth 2 \
            -file $utilization_report

        report_power \
            -hier all \
            -hierarchical_depth 0 \
            -file $power_report

        report_design_analysis \
            -timing \
            -max_paths 5 \
            -file $timing_report

        puts "Generated post-implementation reports:"
        puts "  Utilization: $utilization_report"
        puts "  Power:       $power_report"
        puts "  Timing:      $timing_report"
    }
}

puts "Build finished successfully."
