package hello.hellospring.controller;


import hello.hellospring.domain.AttendanceMember;
import hello.hellospring.service.AttendanceMemberService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Controller
public class AttendanceMemberController {

    private final AttendanceMemberService attendanceMemberService;

    @Autowired
    public AttendanceMemberController(AttendanceMemberService attendanceMemberService){
        this.attendanceMemberService = attendanceMemberService;
    }


    @GetMapping("/attendanceMembers")
    public String list(Model model){
        List<AttendanceMember> attendanceMembers = attendanceMemberService.findMembers();
        model.addAttribute("attendanceMembers", attendanceMembers);
        return "attendanceMembers/attendanceMemberList";

    }

    @GetMapping("/attendanceCheck")
    public String attendanceCheck(){
        return "attendanceCheck/attendanceCheckPage";
    }

    @GetMapping("/setTime")
    public String setTime(@RequestParam(value = "time")int time){
        System.out.println(time);
        String testUrl = "http://127.0.0.1:5000/setTime?time="+time;

        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> response = restTemplate.getForEntity(testUrl, String.class);
        System.out.println(response.getBody());
        return "attendanceCheck/attendanceCheckVideo";
    }

}
