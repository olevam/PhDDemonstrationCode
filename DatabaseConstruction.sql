-- this database has been constructed on postgresSQL

CREATE DATABASE MilanPhD;
-- this is the main table of the database, recording the details of each document
CREATE TABLE alldocuments (
    -- the unique ID for each document analysed in the database
    docid BIGSERIAL PRIMARY KEY, 
    -- the institution of origin for each document, as recorded in the Archivio di Stato of Milan
    origin VARCHAR(100) NOT NULL,
    -- the photo number, as recorded in my filing system
    docnumber VARCHAR(100) NOT NULL,
    -- the type of document, according to my analysis 
    doctype VARCHAR(100) NOT NULL,
    -- the date of the document, according to my analysis
    year date, 
    -- the location of redaction of the document, according to my analysis
    redaction bigint,
    FOREIGN KEY (doctype) REFERENCES doctype(translation),
    FOREIGN KEY (origin) REFERENCES origin(originid))
    ;

-- this  is the table of the actors involved in each document
CREATE TABLE actor(
    -- the unique ID for each actor
    actorid BIGSERIAL PRIMARY KEY,
    -- the activity of the actor
    activity bigint NOT NULL,
    --  the classification of the actor
    classification bigint NOT NULL,
    -- whether the actor is female
    female boolean,
    -- the law of the actor 
    law bigint NOT NULL,
    -- the id of the relevant document
    docid bigint,
    -- the institution of the actor 
    monastery bigint NOT NULL,
    FOREIGN KEY (docid) REFERENCES alldocuments(docid),
    FOREIGN KEY (monastery) REFERENCES monastery(monasteryid),
    FOREIGN KEY (activity) REFERENCES activity(activityid),
    FOREIGN KEY (law) REFERENCES law(lawid),
    FOREIGN KEY (classification) REFERENCES classification(classid));

-- list of types of activities
CREATE TABLE activity(
    -- the unique ID for each activity
    activityid BIGSERIAL PRIMARY KEY,
    -- the name of the activity
    activity VARCHAR(100) NOT NULL UNIQUE);

--list of types of classifications
CREATE TABLE classification(
    -- the unique ID for each classification
    classid BIGSERIAL PRIMARY KEY,
    -- the name of the classification
    classification VARCHAR(100) NOT NULL UNIQUE);

-- list of types of laws
CREATE TABLE law(
    -- the unique ID for each law
    lawid BIGSERIAL PRIMARY KEY,
    -- the law
    law VARCHAR(100) NOT NULL UNIQUE);

-- list of institutions
CREATE TABLE monastery(
    -- the unique ID for each institution
    monasteryid BIGSERIAL PRIMARY KEY,
    -- the name of the institution
    monastery VARCHAR(150) NOT NULL UNIQUE,
    -- the type of institution
    type_institution bigint
    );

-- list of types of institutions
CREATE TABLE type_institution(
    -- unique id for each law
    type_id BIGSERIAL PRIMARY KEY,
    -- the type of institution
    institution VARCHAR(150) NOT NULL UNIQUE);

-- list of types of documents
CREATE TABLE doctype(
    -- unique id for each document type
    id BIGSERIAL PRIMARY KEY,
    -- the document type
    translation VARCHAR(150) NOT NULL UNIQUE);

-- list of origins as found in the archive
CREATE TABLE origin(
    -- unique id for each archival origin 
    originid BIGSERIAL PRIMARY KEY,
    -- the archival origin
    orign VARCHAR(100) NOT NULL UNIQUE);

-- list of locations and coordinates
CREATE TABLE coordinates(
    -- unique id for each coordinate
    coordid BIGSERIAL PRIMARY KEY,
    -- the location name
    locations VARCHAR(100) NOT NULL UNIQUE,
    -- the coordinate geometry for postGIS
    geocoordinates geometry(Point,4326));

-- location of land transacted 
CREATE TABLE land_loc(
    -- unique id for each location
    landlocid BIGSERIAL PRIMARY KEY,
    -- the coordinate id, which connects to the coordinates table
    coordid bigint NOT NULL,
    -- the document id, which connects to the alldocuments table
    docid bigint NOT NULL,
    FOREIGN KEY (coordid) REFERENCES coordinates(coordid),
    FOREIGN KEY (docid) REFERENCES alldocuments(docid));

--  denari exchanged 
CREATE TABLE price(
    -- unique id for each price entry
    priceid BIGSERIAL PRIMARY KEY,
    -- price in denari, in documents recorded in lire, soldi, denari, (1 l = 240d, 1s = 20d)
    price bigint NOT NULL,
    -- what kind of coin, connects to the list of currencies  
    currency bigint NOT NULL,
    -- the document id, which connects to the alldocuments table
    docid bigint NOT NULL,
    FOREIGN KEY (currency) REFERENCES currencies(currid),
    FOREIGN KEY (docid) REFERENCES alldocuments(docid));

-- list of currencies 
CREATE TABLE currencies(
    -- id for each currency
    currid BIGSERIAL PRIMARY KEY,
    -- the currency
    currency VARCHAR(100) NOT NULL UNIQUE);

--  the name of religious leaders in documents 
CREATE TABLE leader(
    -- unique id for each leader entry
    leaderid BIGSERIAL PRIMARY KEY,
    -- first name of leader
    firstname VARCHAR(100) ,
    -- family name of leader
    familyname VARCHAR(100),
    -- the document id, which connects to the alldocuments table
    docid bigint NOT NULL,
    FOREIGN KEY (docid) REFERENCES alldocuments(docid));

--  the name of emissary in documents 
CREATE TABLE emissary(
    -- unique id for each emissary entry
    emisid BIGSERIAL PRIMARY KEY,
    -- first name of emissary
    firstname VARCHAR(100) ,
    -- family name of emissary
    familyname VARCHAR(100),
    -- the document id, which connects to the alldocuments table
    docid bigint NOT NULL,
    FOREIGN KEY (docid) REFERENCES alldocuments(docid));
    


