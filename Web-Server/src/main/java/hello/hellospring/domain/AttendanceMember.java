package hello.hellospring.domain;


import javax.persistence.*;
import java.sql.Timestamp;

//이곳에서 @Table 애노테이션을 통해 DB 테이블명을 바꿔줄 수 있다.
@Entity
@Table(name="attendance")
public class AttendanceMember {


    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String num;

    private String major;

    private String name;

    private int time;

    private double frequency;

    private String result;

    private Timestamp day;

    private String professor;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getNum() {
        return num;
    }

    public void setNum(String num) {
        this.num = num;
    }

    public String getMajor() {
        return major;
    }

    public void setMajor(String major) {
        this.major = major;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getTime() {
        return time;
    }

    public void setTime(int time) {
        this.time = time;
    }

    public double getFrequency() {
        return frequency;
    }

    public void setFrequency(double frequency) {
        this.frequency = frequency;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public Timestamp getDay() {
        return day;
    }

    public void setTimestamp(Timestamp day) {
        this.day = day;
    }

    public String getProfessor() {
        return professor;
    }

    public void setProfessor(String professor) {
        this.professor = professor;
    }
}
