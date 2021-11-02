package hello.hellospring.repository;

import hello.hellospring.domain.AttendanceMember;

import java.util.List;

public interface AttendanceMemberRepository {

    List<AttendanceMember> findAll();
}
