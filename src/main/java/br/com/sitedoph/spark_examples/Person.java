package br.com.sitedoph.spark_examples;

import lombok.Builder;
import lombok.Data;
import lombok.ToString;

/**
 * Criado por ph em 12/14/16.
 */
@Data
@Builder
@ToString
public class Person {
    private long   id;
    private String name;
    private String gender;
    private int    age;
    private String time;
    private String uf;
}
