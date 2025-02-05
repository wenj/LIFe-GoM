var freeviewVideos = [
    ["gt_0006.mp4", "ghg_0006.mp4", "ours_0006.mp4"],
    ["gt_0090.mp4", "ghg_0090.mp4", "ours_0090.mp4"],
    ["gt_0270.mp4", "ghg_0270.mp4", "ours_0270.mp4"],
    ["gt_0426.mp4", "ghg_0426.mp4", "ours_0426.mp4"],
    ["gt_0474.mp4", "ghg_0474.mp4", "ours_0474.mp4"],
    ["gt_0522.mp4", "ghg_0522.mp4", "ours_0522.mp4"]
]

var nvsVideos = [
    ["scene_0008_01_000000_frame_000020.png", "scene_0008_01_000000_frame_000020.mp4"],
    ["scene_0152_01_000050_frame_000018.png", "scene_0152_01_000050_frame_000018.mp4"],
    ["scene_0310_04_000050_frame_000020.png", "scene_0310_04_000050_frame_000020.mp4"]
]

var crossdomainXhumanVideos = [
    ["scene_00016_Take1_f00001_frame_000000_input.png", "scene_00016_Take1_f00001_frame_000000.mp4"],
    ["scene_00041_Take6_f00101_frame_000000_input.png", "scene_00041_Take6_f00101_frame_000000.mp4"],
    ["scene_00025_Take12_frame_000000_input.png", "scene_00025_Take12_frame_000000.mp4"],
    ["scene_00028_Take14_frame_000000_input.png", "scene_00028_Take14_frame_000000.mp4"],
]

var crossdomainPNVideos = [
    ["scene_f3c_input.png", "scene_f3c.mp4"],
    ["scene_m3c_input.png", "scene_m3c.mp4"],
]

var crossdomainDNAVideos = [
    ["scene_0008_01_000000_input.png", "scene_0008_01_000000_frame_000020.mp4"],
    ["scene_0152_01_000050_input.png", "scene_0152_01_000050_frame_000018.mp4"],
    ["scene_0310_04_000050_input.png", "scene_0310_04_000050_frame_000020.mp4"],
]

function ChangeSceneFreeview(idx){
    var li_list = document.getElementById("freeview").children;
    // var m_list = document.getElementById("method-view-ul").children;
    // console.log(idx);
    // console.log(li_list);

    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";

    currentVideos = freeviewVideos[idx]
    document.getElementById("freeview_gt").src = "./medias/freeview/" + currentVideos[0];
    document.getElementById("freeview_ghg").src = "./medias/freeview/" + currentVideos[1];
    document.getElementById("freeview_ours").src = "./medias/freeview/" + currentVideos[2];
}

function ChangeSceneCrossDomainXHuman(idx){
    var li_list = document.getElementById("crossdomain_xhuman").children;
    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    currentVideos = crossdomainXhumanVideos[idx]
    document.getElementById("crossdomain_xhuman_ref").src = "./medias/crossdomain/" + currentVideos[0];
    document.getElementById("crossdomain_xhuman_video").src = "./medias/crossdomain/" + currentVideos[1];
}

function ChangeSceneCrossDomainPN(idx){
    var li_list = document.getElementById("crossdomain_pn").children;
    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    currentVideos = crossdomainPNVideos[idx]
    document.getElementById("crossdomain_pn_ref").src = "./medias/crossdomain/" + currentVideos[0];
    document.getElementById("crossdomain_pn_video").src = "./medias/crossdomain/" + currentVideos[1];
}

function ChangeSceneCrossDomainDNA(idx){
    var li_list = document.getElementById("crossdomain_dna").children;
    for(i = 0; i < li_list.length; i++){
        if (li_list[i].className === "disabled"){
            continue
        }
        li_list[i].className = "";
    }
    li_list[idx].className = "active";
    
    currentVideos = crossdomainDNAVideos[idx]
    document.getElementById("crossdomain_dna_ref").src = "./medias/crossdomain/" + currentVideos[0];
    document.getElementById("crossdomain_dna_video").src = "./medias/crossdomain/" + currentVideos[1];
}