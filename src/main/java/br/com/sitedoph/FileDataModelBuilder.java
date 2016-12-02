package br.com.sitedoph;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.File;
import java.io.IOException;

/**
 * Criado por ph em 12/2/16.
 */
public class FileDataModelBuilder {

    private DataModel dataModel;

    public FileDataModelBuilder(String path) throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        dataModel = new FileDataModel(new File(classLoader.getResource(path).getFile()));
    }

    public DataModel getDataModel() {
        return dataModel;
    }
}
