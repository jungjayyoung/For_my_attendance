package hello.hellospring.repository;

import hello.hellospring.domain.AttendanceMember;

import javax.persistence.EntityManager;
import java.util.List;


public class JpaAttendanceMemberRepository implements AttendanceMemberRepository{

    private final EntityManager em;

    public JpaAttendanceMemberRepository(EntityManager em) {
        this.em = em;
    }

    @Override
    public List<AttendanceMember> findAll() {
        List<AttendanceMember> result = em.createQuery("select m from AttendanceMember m", AttendanceMember.class)
                .getResultList();
        return result;
    }
}
